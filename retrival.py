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

def find_text_indices(text, lookup_table, cat_text, position_idx_record, n=20):
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
    def __init__(self,tokenizer_path:Path='../internlm2-7B', bgem3_list_file_path:str='./bgem3_output/3body.pkl', model_name: str='BAAI/bge-m3'):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.splitter = ChineseRecursiveTextSplitter()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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

    def preprocess_data(self, chunk_size=250, chunk_overlap=0, data_input_dir:Path='./input_dir', output_path:Path='./bgem3_output/', tokenized_text_path:Path='./output_datas/tokenized_prompt/'):
        # preprocessing data, split text into chunks and encode them
        if not os.path.notexists(data_input_dir):
            # 
            raise FileNotFoundError(f"File {data_input_dir} not found")
        if not os.path.exists(tokenized_text_path):
            # tokenize text
            self.tokenize_file(data_input_dir, tokenized_text_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        text_splitter = ChineseRecursiveTextSplitter(
            keep_separator=True,
            is_separator_regex=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # walk through all files in input_data
        for root, _, files in os.walk(data_input_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(Path(root) / file, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # split text into chunks
                        look_up_table, cat_text, position_idx_record = read_lookup_table(tokenized_text_path/file)
                        processed_chunk_list = []
                        not_found_list = []
                        chunks = text_splitter.split_text(text)
                        for i, chunk in enumerate(chunks):
                            meta = self.template.copy()
                            start, end = find_text_indices(chunk, look_up_table, cat_text, position_idx_record)
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
                            meta['doc_id'] = file
                            processed_chunk_list.append(meta)
                    # save to pickle
                    # exclude .txt tail
                    with open(output_path / f'{file[:-4]}.pkl', 'wb') as f:
                        pickle.dump(processed_chunk_list, f)
                    
                    with open(output_path / 'not_found.txt', 'w', encoding='utf-8') as f:
                        for chunk in not_found_list:
                            f.write(chunk + '\n')
        

    def retrive_data(self, queries, top_k=5, expand_length=0, is_merge_overlaped=False, mode='colbert+lexical_weights+dense'):
        # 对queries进行检索
        score_list = self.compute_score(queries)
        queries_top_k_idx = self.get_top_k_idx(score_list, top_k, mode)
        queries_meta_list = []
        for i, top_k_idx_list in enumerate(queries_top_k_idx):
            query_meta_list = []
            for idx in top_k_idx_list:
                meta = {}
                meta['content'] = self.bgem3_list[idx]['content']
                meta['start'] = self.bgem3_list[idx]['start']
                meta['end'] = self.bgem3_list[idx]['end']
                # meta['doc_id'] = self.bgem3_list[idx]['doc_id']
                meta['query'] = queries[i]
                query_meta_list.append(meta)
            queries_meta_list.append(query_meta_list)
        return queries_meta_list
    
    def get_top_k_idx(self, query_score_list, k=5, mode='colbert+lexical_weights+dense'):
        # 对每个query进行embedding的打分计算
        query_score_top_k_idx = []
        for query_score in query_score_list:
            query_score_top_k_idx.append(
                sorted(range(len(query_score)), key=lambda i: query_score[i][mode], reverse=True)[:k]
            )
        return query_score_top_k_idx

        
    def compute_score(self, queries:list[str], query_batch_size:int=256, weights_for_different_modes: list[float] = None):        
        query_score_list = []
        for query in queries:
            score_list = []
            query_output = self.model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=True)

            for sentence in self.bgem3_list:
                dense_score = query_output['dense_vecs']@ sentence['dense_vecs'].T
                sparse_score = self.model.compute_lexical_matching_score(query_output['lexical_weights'], sentence['lexical_weights'])
                colbert_score = self.model.colbert_score(query_output['colbert_vecs'], sentence['colbert_vecs'])
                if weights_for_different_modes is None:
                    weights_for_different_modes = [1, 1., 1.]
                    weight_sum = 3
                    print("default weights for dense, lexical_weights, colbert are [1.0, 1.0, 1.0] ")
                else:
                    assert len(weights_for_different_modes) == 3
                    weight_sum = sum(weights_for_different_modes)
                
                all_scores = {}
                all_scores['dense'] = dense_score
                all_scores['lexical_weights'] = sparse_score
                all_scores['colbert'] = colbert_score
                all_scores['lexical_weights+dense'] = (sparse_score * weights_for_different_modes[1] + dense_score * weights_for_different_modes[0])/(weights_for_different_modes[1]+weights_for_different_modes[0])
                all_scores['colbert+lexical_weights+dense']= (sparse_score * weights_for_different_modes[1] + dense_score * weights_for_different_modes[0] + colbert_score * weights_for_different_modes[2])/weight_sum
                score_list.append(all_scores)
            query_score_list.append(score_list)
        return query_score_list