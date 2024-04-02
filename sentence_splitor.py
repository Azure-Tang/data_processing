import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)
import json

def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]


def find_subsequence_start(sub, seq):
    # 将子序列和序列转换为列表
    sub_list = sub.tolist()
    seq_list = seq.tolist()

    # 遍历序列，查找子序列的起始下标
    for i in range(len(seq_list) - len(sub_list) + 1):
        if seq_list[i:i + len(sub_list)] == sub_list:
            return i  # 返回子序列的起始下标
    return -1  # 如果未找到子序列，返回-1

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('../internlm2-7B', trust_remote_code=True)
    chunk_size = 250
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=0
    )
    
    for doc_id in range(128):
        doc_context = ''
        f = open(f"../gpt-fast/book_prompt/{doc_id}.txt")
        for line in f:
            doc_context += line
        f.close()
        doc_json = []
        doc_encoded = tokenizer(doc_context, return_tensors='pt').input_ids[0][1:]
        chunks = text_splitter.split_text(doc_context)
        for chunk_id, chunk in enumerate(chunks):
            chunk_encoded = tokenizer(chunk, return_tensors='pt').input_ids[0][1:]
            # print(chunk_id)
            # print(chunk)
            # print(chunk_encoded)
            start = find_subsequence_start(chunk_encoded[:5], doc_encoded)
            if start == -1:
                print('error')
            end = start + chunk_encoded.shape[0]
            chunk_json = {}
            chunk_json['content'] = chunk
            chunk_json['start'] = start
            chunk_json['end']  = end
            doc_json.append(chunk_json)
        json.dump(doc_json, open(f"../gpt-fast/book_prompt/{doc_id}_{chunk_size}.json", 'w'), ensure_ascii=False, indent=4)