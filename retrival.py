from pathlib import Path
import os
import sys
import re
import json
from sentence_splitor import ChineseRecursiveTextSplitter
import pickle
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.BGE_M3 import BGEM3ForInference
import torch

def find_and_expand(target_str, folder_path, expansion_length=1000, start_ignore=0, end_ignore=0):
    # 将目标字符串转换为一个正则表达式，允许在其字符之间存在最多为n的空白字符
    target_regex = re.compile(".{0,10}".join(map(re.escape, target_str)), re.DOTALL)

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    assert len(content) >= start_ignore + end_ignore, f"File {file_path} is too short"
                    content = content[start_ignore:-end_ignore]
                    # 首先尝试直接搜索目标字符串
                    index = content.find(target_str)
                    # 如果直接搜索失败，则尝试使用正则表达式搜索
                    if index == -1:
                        match = target_regex.search(content)
                        if match:
                            index = match.start()
                    # 如果找到了匹配项, 无论是直接搜索还是正则搜索
                    if index != -1:
                        start = max(0, index - expansion_length // 2)
                        end = min(
                            len(content),
                            index + len(target_str) + expansion_length // 2,
                        )
                        # 找到开始和结束点附近的分隔符，以保持句子完整性
                        start = content.rfind("\n", 0, start)
                        if start == -1:
                            start = 0
                        else:
                            start += 1
                        end = content.find("\n", end)
                        if end == -1:
                            end = len(content) - 1
                        expanded_str = content[start:end]
                        return expanded_str, file
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Target string not found in any file. {target_str}")
    return None, None

def read_lookup_table(file_path):
    lookup_table = []
    # 定义正则表达式来匹配 idx, token_id, 和可能跨越多行的 token_string
    pattern = re.compile(r"(\d+),\s*(\d+),\s*'((?:[^']|'(?!$))*)'", re.MULTILINE)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        # 使用正则表达式匹配所有行
        matches = pattern.findall(content)
        for match in matches:
            idx, token_id, token_string = match
            # 替换代表换行符的单引号
            token_string = token_string.replace("\n'", "\n")
            lookup_table.append((int(idx), int(token_id), token_string))
    cat_text = ""
    position_idx_record = [] # 记录每个char在lookup_table中的idx
    for idx, _, token_string in lookup_table:
        cat_text += token_string
        position_idx_record+=[idx]*len(token_string)
    return lookup_table, cat_text, position_idx_record

def find_text_indices(text, cat_text, position_idx_record, n=20):
    # 取巧只匹配段落的前20个和后20个字符来定位整段的位置
    # 初始化 start_idx 和 end_idx 为 None
    start_idx = end_idx = None

    # 通过头，尾各取20个字符，查找text在cat_text中的位置
    start_txt = text[:n]
    end_txt = text[-n:]

    start_idx = cat_text.find(start_txt)
    # 如果直接搜索失败，则尝试使用正则表达式搜索
    if start_idx == -1:
        target_regex = re.compile(".{0,10}".join(map(re.escape, text)), re.DOTALL)
        match = target_regex.search(cat_text)
        if match:
            start_idx = match.start()
    
    end_idx = cat_text.find(end_txt) + len(end_txt) - 1
    # 如果直接搜索失败，则尝试使用正则表达式搜索
    if end_idx == -1:
        target_regex = re.compile(".{0,10}".join(map(re.escape, text)), re.DOTALL)
        match = target_regex.search(cat_text)
        if match:
            end_idx = match.end()
    
    if start_idx != -1 and end_idx != -1:
        start_idx = position_idx_record[start_idx]
        end_idx = position_idx_record[end_idx]

        return start_idx, end_idx

    else:
        # raise Exception(f"Failed to find text: {text[:10]}")
        print(f"Failed to find text: {text[:n]}")
        return text, None

def tokenize_file(tokenizer, text_path:Path, output_path:Path=Path('./output_datas/tokenized_prompt/')):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, dirs, files in os.walk(text_path):
        for file in files:
            if file.endswith('.txt'):
                with open(Path(root) / file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False)
                    token_ids = encoded_input['input_ids'][0]
                    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]

                with open(output_path / file, 'w', encoding='utf-8') as file:
                    for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
                        file.write(f"{i}, {token_id}, '{token}'\n")


def compute_similarity(q_reps, p_reps):
    if len(p_reps.size()) == 2:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))

def cal_dense_score(q_reps, p_reps, temperature=1.0):
    scores = compute_similarity(q_reps, p_reps) / temperature
    scores = scores.view(q_reps.size(0), -1)
    return scores

def cal_sparse_score(q_reps, p_reps, temperature=1.0):
    scores = compute_similarity(q_reps, p_reps) / temperature
    scores = scores.view(q_reps.size(0), -1)
    return scores

def cal_colbert_score(q_reps, p_reps, q_mask: torch.Tensor, temperature=1.0):
    token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
    scores, _ = token_scores.max(-1)
    scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
    scores = scores / temperature
    return scores

class retrival_model:
    def __init__(self,tokenizer_path:Path='../internlm2-7B', bgem3_list_file_path:str='./bgem3_output/3body.txt.pkl', model_name: str='BAAI/bge-m3', use_dense_and_lexical_score=True, use_colbert_score=True, use_fp16=True, book_dir:Path='./book_dir', middle_files_dir:Path='./datas'):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.splitter = ChineseRecursiveTextSplitter()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if not use_dense_and_lexical_score and not use_colbert_score:
            raise ValueError("At least one of use_dense_and_lexical_score and use_colbert_score should be True")
        self.use_dense_and_lexical_score = use_dense_and_lexical_score
        self.use_colbert_score = use_colbert_score
        

        if not os.path.exists(book_dir):
            raise FileNotFoundError(f"File {book_dir} not found")
        self.book_dir = book_dir
        if not os.path.exists(middle_files_dir):
            raise FileNotFoundError(f"File {middle_files_dir} not found")
        self.middle_files_dir = middle_files_dir
        

        if not os.path.exists(bgem3_list_file_path):
            raise FileNotFoundError(f"File {bgem3_list_file_path} not found")
        with open(bgem3_list_file_path, "rb") as file:
            self.bgem3_list = pickle.load(file)

        self.template={
            "content": "text",
            "start": 0,
            "end": 10,
            "dense_vecs": "matrix",
            "sparse_vecs": "matrix",
            "colbert_vecs": "matrix"
        }

    def preprocess_data(self, chunk_size=250, chunk_overlap=0, book_dir:Path='./input_dir', output_path:Path='./datas/bgem3_output/', middle_files_dir:Path='./datas/tokenized_prompt/'):
        # preprocessing data, split text into chunks and encode them
        if not os.path.notexists(book_dir):
            raise FileNotFoundError(f"File {book_dir} not found")
        if not os.path.exists(middle_files_dir):
            # tokenize text
            self.tokenize_file(book_dir, middle_files_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        text_splitter = ChineseRecursiveTextSplitter(
            keep_separator=True,
            is_separator_regex=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # walk through all files in input_data
        for root, _, files in os.walk(book_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(Path(root) / file, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # split text into chunks
                        _, cat_text, position_idx_record = read_lookup_table(middle_files_dir/file)
                        processed_chunk_list = []
                        not_found_list = []
                        chunks = text_splitter.split_text(text)
                        for i, chunk in enumerate(chunks):
                            meta = self.template.copy()
                            start, end = find_text_indices(chunk, cat_text, position_idx_record)
                            if end == None:
                                not_found_list.append(chunk)
                                continue
                            output = self.model.encode(chunk, return_dense=True, return_sparse=True, return_colbert_vecs=True)
                            meta['content'] = chunk
                            meta['start'] = start
                            meta['end'] = end
                            meta['dense_vecs'] = output['dense_vecs']
                            meta['sparse_vecs'] = output['sparse_vecs']
                            meta['colbert_vecs'] = output['colbert_vecs']
                            meta['doc'] = file
                            processed_chunk_list.append(meta)
                    # save to pickle
                    # exclude .txt tail
                    with open(output_path / f'{file[:-4]}.pkl', 'wb') as f:
                        pickle.dump(processed_chunk_list, f)
                    
                    with open(output_path / 'not_found.txt', 'w', encoding='utf-8') as f:
                        for chunk in not_found_list:
                            f.write(chunk + '\n')

    def expand_text(self, queries_meta_list:list[dict], expand_length=0):
        expanded_queries_meta_list = []
        for query_meta_list in queries_meta_list:
            expanded_query_meta_list = []
            for sentence_meta in query_meta_list:
                expanded_text, _ = find_and_expand(sentence_meta['content'], self.book_dir, expansion_length=expand_length)
                if expanded_text is None:
                    print(f"Failed to expand text: {sentence_meta['content']}")
                    expanded_query_meta_list.append(sentence_meta)
                    continue
                _, cat_text, position_idx_record = read_lookup_table(self.middle_files_dir/sentence_meta['doc'])
                start, end = find_text_indices(expanded_text, cat_text, position_idx_record)
                output = self.model.encode(expanded_text, return_dense=True, return_sparse=True, return_colbert_vecs=True)
                expended_sentence_meta = {}
                expended_sentence_meta['content'] = expanded_text
                expended_sentence_meta['start'] = start
                expended_sentence_meta['end'] = end
                expended_sentence_meta['doc'] = sentence_meta['doc']
                expended_sentence_meta['query'] = sentence_meta['query']
            expanded_queries_meta_list.append(expanded_query_meta_list)
        return expanded_queries_meta_list

    def retrive_data(self, queries, top_k=5, expand_length=0, is_merge_overlaped=False):
        # 对queries进行检索
        score_list = self.compute_score(queries)
        queries_top_k_idx = self.get_top_k_idx(score_list, top_k)
        queries_meta_list = []
        for i, top_k_idx_list in enumerate(queries_top_k_idx):
            query_meta_list = []
            for idx in top_k_idx_list:
                sentence_meta = {}
                sentence_meta['content'] = self.bgem3_list[idx]['content']
                sentence_meta['start'] = self.bgem3_list[idx]['start']
                sentence_meta['end'] = self.bgem3_list[idx]['end']
                sentence_meta['doc'] = self.bgem3_list[idx]['doc']
                sentence_meta['query'] = queries[i]
                query_meta_list.append(sentence_meta)
            queries_meta_list.append(query_meta_list)
        if expand_length != 0:
            queries_meta_list = self.expand_text(queries_meta_list, expand_length)

        return queries_meta_list
    
    def get_top_k_idx(self, query_score_list, k=5):
        # 对每个query进行embedding的打分计算
        query_score_top_k_idx = []
        for query_score in query_score_list:
            query_score_top_k_idx.append(
                sorted(range(len(query_score)), key=lambda i: query_score[i], reverse=True)[:k]
            )
        return query_score_top_k_idx

        
    def compute_score(self, queries:list[str], query_batch_size:int=256, weights_for_different_modes: list[float] = None):        
        queries_score_list = []
        for query in queries:
            score_list = []
            query_output = self.model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=True)

            for sentence in self.bgem3_list:
                if self.use_dense_and_lexical_score:
                    dense_score = query_output['dense_vecs']@ sentence['dense_vecs'].T
                    lexical_weights_score = self.model.compute_lexical_matching_score(query_output['lexical_weights'], sentence['lexical_weights'])
                if self.use_colbert_score:
                    colbert_score = self.model.colbert_score(query_output['colbert_vecs'], sentence['colbert_vecs'])
                
                if weights_for_different_modes is None:
                    weights_for_different_modes = [1, 1., 1.]
                    weight_sum = 0
                    if self.use_dense_and_lexical_score == True:
                        weight_sum += 2
                    if self.use_colbert_score == True:
                        weight_sum += 1
                    print("default weights for dense, lexical_weights, colbert are [1.0, 1.0, 1.0] or [1.0, 1.0] ")
                else:
                    if self.use_dense_and_lexical_score:
                        weight_sum += sum(weights_for_different_modes[:2])
                    if self.use_colbert_score:
                        weight_sum += weights_for_different_modes[2]
                all_score = 0
                if self.use_dense_and_lexical_score:
                    all_score += dense_score * weights_for_different_modes[0] + lexical_weights_score * weights_for_different_modes[1]

                if self.use_colbert_score:
                    all_score += colbert_score * weights_for_different_modes[2]

                all_score = all_score/weight_sum 

                score_list.append(all_score)
            queries_score_list.append(score_list)
        return queries_score_list