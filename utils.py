import re
import json
import datasets
import nltk
import pandas as pd
import numpy as np
import torch

from hanspell import spell_checker
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 편집거리 및 후처리 간 사용할 형태소 종류입니다.
__TAGSETS_TO_USE = ['NNP','NNG','NNB','NNBC','NR','NP','SH','SL','SN']

def split_by_end(inputs : str) -> str :
    """
    문장을 마침표 단위로 분할합니다.
    별도로 메소드를 만든 이유는 문장에 소수점이 섞인 경우 이를 기준으로 나뉘기 때문에
    마침표가 소수점일 경우 다시 붙여주기 위함입니다.
    기타 문장부호(물음표, 느낌표 등)는 우선은 생략하고 진행했습니다.

    inputs : 문자열 하나입니다. Context 문장을 입력받습니다.
    return : Context 문장을 마침표 기준으로 분할해 여러 개의 문장이 담긴 리스트를 반환합니다. (문자열 리스트)

    """
    splitted = []
    inputs = inputs.split('.')

    for i in inputs :
        if splitted and i and splitted[-1][-1].isdigit() and i[0].isdigit() :
            splitted[-1] = splitted[-1] + '.' + i
        else :
            splitted.append(i)

    return splitted

def cleaning(inp : str) -> str :

    """
    전처리 메소드입니다.
    일반적인 특수문자들과 공백 등을 없애는 간단한 논리만 가지고 있습니다.
    전처리 결과 문장의 첫 시작이 구두점인 경우가 있어서 특이 케이스에 대한 논리도 아래 if문에 작성되어있습니다.
    또한 뉴스 기사 데이터이기 때문에 몇몇 Context의 경우 마지막에 XXX기자 또는 XXX 특파원 등의 의미 없는 정보들이 있어 이런 것도 지우고자 했습니다.

    inputs : 전처리 대상 문자열입니다.
    return : 전처리를 마친 문자열을 반환합니다.
    """
    inp = re.sub("\s{2,}", ' ', inp)
    inp = re.sub("\.{2,}", '.', inp)
    inp = inp.strip()

    if inp.startswith(".") :
        while inp.startswith(".") :
            inp = inp[1:]

    if inp.endswith("기자") or inp.endswith("특파원") :
        inp_list = inp.split(".")
        inp = '.'.join(inp_list[:-1]) + '.'
    return inp


def load_data(filepath : str, do_preprocessing : bool = False) -> pd.DataFrame :
    """
    기본 데이터(KLUE QA)의 경로를 입력받아 pandas DataFrame을 반환합니다.
    
    filepath : 사용할 json 데이터 경로를 입력합니다.
    return : json 파일을 parsing해 pandas DataFrame으로 반환합니다.
    """

    temp_data = defaultdict(list)

    with open(filepath) as raw :
        readed = json.load(raw)

    for d in tqdm(readed["data"]) :
        for p in d["paragraphs"] :
            for q in p['qas'] :
                if q["answers"] :
                    for a in q["answers"] :
                        temp_data['guid'].append(q["guid"])
                        context = p["context"].lower()                              # 모든 영문자를 소문자로 바꿉니다.
                        start_idx = a['answer_start']                               # 정답 문자열의 시작 위치를 가져옵니다.
                        answer = a["text"]
                        if do_preprocessing :
                            preprocessed_before_start = cleaning(context[:start_idx])   # 정답 문자열의 시작 위치를 기준으로 문장을 반으로 잘라 절반 앞부분에 대한 전처리를 합니다.
                            preprocessed_after_start = cleaning(context[start_idx:])    # 위에서 나눈 나머지 뒷 부분에 대한 전처리를 합니다.
                            start_idx = len(preprocessed_before_start)                  # 정답 문자열은 반으로 자른 뒷 부분의 첫 시작이 됩니다. 따라서 정답 문자열의 시작 인덱스는 앞 부분의 전처리한 문자열의 길이가 됩니다.
                            answer = cleaning(a['text'].lower())                # 정답 문자열까지 전처리를 합니다. 처리하지 않았더니 인덱스가 조금씩 꼬이게 되어서 일단은 인덱스를 맞춰줍니다... 추후 좋은 방안이 생기면 수정해보아요!
                            context = preprocessed_before_start + preprocessed_after_start  # 위에서 반반 나누어 전처리를 했던 원문을 붙입니다.
                        
                        temp_data['context'].append(context)                            # 붙인 원문을 저장합니다.
                        temp_data['question'].append(q["question"])                 # 질문을 저장합니다.
                        temp_data['answer'].append(answer)                          # 정답을 저장합니다.
                        temp_data['answer_start'].append(start_idx)                         # 정답의 시작 인덱스를 저장합니다.

                else :
                    temp_data['guid'].append(q["guid"])
                    context = p["context"].lower()                              # 모든 영문자를 소문자로 바꿉니다.
                    context = cleaning(context)
                    temp_data['context'].append(context)                            # 붙인 원문을 저장합니다.
                    temp_data['question'].append(q["question"].lower())                 # 질문을 저장합니다.

    data = pd.DataFrame.from_dict(temp_data)
    if "answer_start" in data.columns :
        data.loc[:, "answer_end"] = data.loc[:, "answer_start"] + data.answer.str.len()       # 정답이 끝나는 인덱스는 정답의 시작 위치 + 정답 문자열의 길이로 계산이 가능합니다.
    data = data.drop_duplicates()                                                                   # 중복되는 행을 제거합니다.
    data = data.reset_index(drop = True)
    return data

def load_external_data(filepath : str) -> pd.DataFrame :
    """
    AiHub에 공개된 외부 데이터를 끌어올 때 사용 가능한 함수입니다.
    기존 데이터와 형식이 달라 별도로 작성했습니다.
    사용은 load_data와 같습니다.
    """

    external_temp = defaultdict(list)

    with open(filepath) as raw :
        readed = json.load(raw)
        for d in tqdm(readed["data"]) :
            for p in d["paragraphs"] :
                for q in p['qas'] :
                        context = p["context"].lower()                              # 모든 영문자를 소문자로 바꿉니다.
                        start_idx = q["answers"]['answer_start']                               # 정답 문자열의 시작 위치를 가져옵니다.
                        preprocessed_before_start = cleaning(context[:start_idx])   # 정답 문자열의 시작 위치를 기준으로 문장을 반으로 잘라 절반 앞부분에 대한 전처리를 합니다.
                        preprocessed_after_start = cleaning(context[start_idx:])    # 위에서 나눈 나머지 뒷 부분에 대한 전처리를 합니다.

                        start_idx = len(preprocessed_before_start)                  # 정답 문자열은 반으로 자른 뒷 부분의 첫 시작이 됩니다. 따라서 정답 문자열의 시작 인덱스는 앞 부분의 전처리한 문자열의 길이가 됩니다.

                        cleaned_answer = cleaning(q["answers"]['text'].lower())                # 정답 문자열까지 전처리를 합니다. 처리하지 않았더니 인덱스가 조금씩 꼬이게 되어서 일단은 인덱스를 맞춰줍니다... 추후 좋은 방안이 생기면 수정해보아요!

                        new_context = preprocessed_before_start + preprocessed_after_start  # 위에서 반반 나누어 전처리를 했던 원문을 붙입니다.
                        external_temp['context'].append(new_context)                            # 붙인 원문을 저장합니다.
                        external_temp['question'].append(q["question"].lower() + '?')                 # 질문을 저장합니다.
                        external_temp['answer'].append(cleaned_answer)                          # 정답을 저장합니다.
                        external_temp['answer_start'].append(start_idx)                         # 정답의 시작 인덱스를 저장합니다.

    external_train = pd.DataFrame.from_dict(external_temp)
    external_train.loc[:, "answer_end"] = external_train.loc[:, "answer_start"] + external_train.answer.str.len()       # 정답이 끝나는 인덱스는 정답의 시작 위치 + 정답 문자열의 길이로 계산이 가능합니다.
    external_train = external_train.drop_duplicates()                                                                   # 중복되는 행을 제거합니다.
    external_train = external_train.reset_index(drop = True)
    return external_train

def sampling_context_with_cosine_similarity(inputs : pd.DataFrame, tokenizer, tagger) :

    """
    해당 메소드의 예상 소요시간은 Colab Pro+ 기준 약 2분입니다.
    Context 내 문장들과 Question의 유사도를 측정하기 위해 TF-IDF 기반의 Cosine Similarity를 사용합니다.
    https://konlpy.org/ko/latest/morph/#comparison-between-pos-tagging-classes (형태소 분석기 성능 비교)
    형태소 분석기 별로 성능 분석을 하려 했지만 생각보다 parsing에 시간이 어마어마하게 오래걸려서 Mecab 외에는 현실적으로 이용하기 어려울 것으로 예상합니다.

    inputs : context를 question 기준으로 resampling할 pandas DataFrame
    tokenizer : pretrained model에서 사용한 tokenizer
    tagger : konlpy의 형태소 분석기

    returns : resampling된 context를 가진 pandas DataFrame, 정답 문장을 포함하지 않는 train example의 index
    """
    n_mismatch = 0
    similarity_failed = 0
    n_correct = 0
    n_adjust = 0
    rows_to_drop = []

    # TF-IDF와 Cosine-similarity를 기반으로 주어진 Context에서 Question과 가장 유사한 문장을 기준으로 주변 문장을 새로 샘플링합니다.
    # 이 때 가장 유사한 문장을 중심으로 주변 문장들을 들고오기 때문에 원문 Context의 가장자리(처음 또는 끝)에 위치한 문장일 수록 샘플링 될 가능성이 낮아질 것입니다.
    # Question과 새로 샘플링된 Context의 토큰화 이후 길이를 최대한 512로 맞추려고 했습니다. (유사도가 높다고 해서 정답 문장이 꼭 속해있다는 보장은 없기 때문에 최대한 많이 들고오려고 했습니다.)

    for i, (question, context) in enumerate(zip(inputs.question, inputs.context)) :
        tokenized = tokenizer(question, context)
        if len(tokenized["input_ids"]) > tokenizer.model_max_length : # 토큰화 이후 길이가 512 이상인 문장에 대해서만 작동합니다.
            n_adjust += 1

            tfidf_inputs = [' '.join(tagger.morphs(question.strip()))] + [' '.join(tagger.morphs(c.strip())) for c in split_by_end(context) if c != ''] # Mecab을 이용해 형태소 단위로 분할하고 이를 TF-IDF의 입력으로 사용합니다.
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(tfidf_inputs)                    # TF-IDF는 형태소 단위로 띄어쓰기가 된 Context와 Question을 입력으로 받습니다.
            cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)       # 현재 Context와 Question의 Cosine similarity를 계산합니다.
            
            center = np.argmax(cosine_matrix[0][1:])                        # Context의 문장 중 Question과 유사도가 가장 높은 문장의 인덱스를 가져옵니다. (이하 center라고 하겠습니다.)
            left = center - 1                                               # center를 기준으로 이전 이후 문장을 반복적으로 가져옵니다. left는 center 이전 문장, right는 center 뒤 문장에 해당됩니다.
            right = center + 1
            context_sentences = split_by_end(context)

            fix_left = False
            fix_right = False

            while len(tokenizer(question, '.'.join(context_sentences[left:right]) + '.')['input_ids']) <= tokenizer.model_max_length : # context에서 left~right의 범위에 속하는 문장들과 question 문장을 토큰화했을 때 길이가 512가 넘어갈 때 까지 left와 right의 범위를 늘려갑니다.
                left = max(0, left - 1)
                right = min(len(context_sentences) - 1, right + 1)
                if left == 0 and right == len(context_sentences) - 1 :              # 무한루프를 방지하기 위한 코드입니다. 토큰 길이가 512에 딱 맞는 경우 계속 돌게 돼서 이걸 막기 위해 빠져나오는 논리를 추가했습니다.
                    break

            if left == 0 :                                      # 지금 들고온 문장은 512가 넘어가기 때문에 이걸 처리하는 과정이 필요합니다.
                fix_left = True                                 # 전반적인 과정은 마지막으로 추가된 left과 right 중 상대적으로 center와 멀리 있는 문장을 제거하는 것으로 진행됩니다.
            if right == len(context_sentences) - 1 :            # fix_left는 left가 첫 번째 문장까지 뻗어갔을 경우 left가 첫 번째 문장에 해당한다는 것을 표시하기 위해 작성했습니다.
                fix_right = True                                # fix_right는 right가 마지막 문장까지 뻗어갔을 경우 right가 마지막 문장에 해당한다는 것을 표시하기 위해 작성했습니다.

            if (fix_left and fix_right) or (not fix_left and not fix_right) :       # fix_left와 fix_right가 둘 다 True일 경우 위 과정을 거쳐 들고운 문장이 원문 context와 같다는 의미입니다. 따라서 둘 중 하나를 제거해야 합니다.
                left_dist = abs(center - left)                                  # fix_left와 fix_right가 둘 다 False일 경우 context의 처음과 끝 문장까지 뻗어나가지 못했다는 뜻입니다. 어찌되었건 둘 중 하나를 제거해야 합니다.
                right_dist = abs(center - right)                                # 둘 중 하나를 제거하기 위해 left와 right 중 상대적으로 멀리 있는 문장을 제거합니다.
                if left_dist <= right_dist :                                        # 만일 둘 다 center로부터 같은 거리에 떨어져 있다면, 일반적으로 신문기사가 두괄식으로 작성되어있다는 점에 착안해 뒤에 있는 문장을 제거하고자 했습니다.
                    right -= 1                                                      # 위 가정이 완벽하진 않은 듯 합니다. 실제로 이 과정으로부터 뽑아온 마지막 문장의 바로 뒤에 정답을 가지고 있는 문장이 있어서 항상 옳게 작동하지는 않습니다...
                else :
                    left += 1
            elif fix_left :             # 만일 left가 첫 번째 문장이라면 위 while문에서 문장의 뒷 방향으로만 뻗어나가며 문장을 가져왔을겁니다. 따라서 뒤 문장을 제거합니다.
                right -= 1
            elif fix_right :            # right가 마지막 문장일 경우 위와 같은 흐름으로 앞 문장을 제거합니다.
                left += 1

            new_context = '.'.join(context_sentences[left:right]) + '.'                     # 이렇게 유사도 기반으로 뽑은 문장을 다시 원래 형태로 복원합니다.
            
            if "answer" in inputs.columns :
                diff = len('.'.join(context_sentences[:left]) + '.') if left != 0 else 0        # 원래 문장을 수정했기 때문에 정답 단어가 출현하는 index가 또다시 발목을 잡습니다.
                new_start_idx = inputs.loc[i, "answer_start"] - diff                    # left 이전에 있는 문장들의 길이를 정답 단어의 인덱스에서 빼면 뽑아온 문장에서 정답 단어의 인덱스를 찾아낼 수 있으니 이렇게 처리했습니다.
                new_end_idx = inputs.loc[i, "answer_end"] - diff

                if inputs.loc[i, "answer"] == new_context[new_start_idx:new_end_idx] :  # 새로 뽑아온 문장으로부터 정답을 추출했을 때의 결과가 원래 정답과 같을 경우 문제가 없으니 데이터를 갱신합니다.
                    n_correct += 1                                                              # 정상적으로 들고 올 수 있는 문장의 개수를 셉니다.
                    inputs.loc[i, "context"] = new_context
                    inputs.loc[i, "answer_start"] = new_start_idx
                    inputs.loc[i, "answer_end"] = new_end_idx

                if new_start_idx < 0 or len(new_context) < new_start_idx:                       # 여기서부터는 문장을 유사도 기반으로 새로 추출한 것의 완성도를 평가하고자 작성했습니다.
                    similarity_failed += 1                                                   # 정답이 뽑아온 문장에 속해있지 않을 때 유사도 기반으로 문장을 추출한 것이 실패한 경우가 될 것입니다. 이를 similarity_failed로 count했습니다.
                    rows_to_drop.append(i)
                elif inputs.loc[i, "answer"] != new_context[new_start_idx:new_end_idx] :    # 정답이 뽑아온 문장에 속해있긴 하지만 알 수 없는 이유로 지금 들고 온 문장에서 정답에 접근할 수 없는 경우를 셌습니다.
                    n_mismatch += 1                                                                 # rows_to_drop은 이렇게 문제가 발생한 케이스의 인덱스를 저장하는 리스트입니다. 아래에서 이걸 제외하고 학습시키려고 했습니다.
                    rows_to_drop.append(i)
        else :
            n_correct += 1              # 위 과정은 토큰 길이가 512를 넘었을 때 동작합니다. 위 케이스에 해당되지 않는 경우는 항상 옳은 문장과 답을 가져올 수 있기 때문에 정상적으로 들고 올 수 있는 문장으로 간주합니다.    print("512개 토큰이 넘는 문장의 비율 :", n_adjust / len(inputs))
    if "answer" in inputs.columns :
        print("코드 오류로 추정되는 문제로 인해 사용하지 못하는 문장의 비율 :", n_mismatch / len(inputs))
        print("유사도 기반으로 추출한 문장에 정답 단어가 없는 문장의 비율 :", similarity_failed / len(inputs))
        print("위 두 문제를 합한 전체 오류 :", (n_mismatch + similarity_failed) / len(inputs))
        print("실제로 유사도 기반 추출을 한 문장들 중 오류가 있는 문장의 비율:", (n_mismatch + similarity_failed) / n_adjust)
        print("전체 문장들 중 사용할 수 있는 문장의 비율 :", n_correct / len(inputs))

    return inputs, rows_to_drop

def __tokenizing(inputs, tokenizer, training):
    model_inputs = tokenizer(inputs["question"], inputs["context"], truncation = "only_second", return_offsets_mapping = True)          # 모델에 들어가는 입력은 [질문, 본문]으로 들어가게 됩니다. 여기서 질문은 잘리면 치명적일테니 본문을 자릅니다.
                                                                                                                                            # 정답의 시작 위치는 문자열 기준으로 표현되어있기 때문에 토큰 기준의 시작 위치를 알아내야 합니다.
                                                                                                                                            # return_offsets_mapping = True로 반환되는 결과로 토큰과 문자열의 인덱스를 매핑시킬 수 있기 때문에
                                                                                                                                            # 이 결과를 끌고 와서 문자열과 토큰 인덱스를 연결지어 토큰 기준의 정답 시작 위치로 변환합니다. 
    if training :
        model_inputs["golden_answer"] = tokenizer(inputs["answer"], padding = "max_length", max_length = 128)["input_ids"]
        start_positions = []            # 토큰 기준 정답 시작 위치를 저장할 리스트입니다.
        end_positions = []              # 토큰 기준 정답 끝 위치를 저장할 리스트입니다.
        for i, offset in enumerate(model_inputs["offset_mapping"]) :    # 이 함수는 batch 단위로 돌아갈거라 별도 파라미터를 조정하지 않는다면 model_inputs에 있는 문장 개수가 1000개일 것입니다.
                                                                        # 따라서 1000개 문장에 대해 정답의 토큰 기준 시작 위치와 끝 위치를 알아내야 합니다.
                                                                        # offset_mapping은 위 return_offsets_mapping = True로 반환된 결과입니다. 이것도 모든 문장에 대해 있을테니 이걸 기준으로 순회합니다.

            context_info = np.where(np.array(model_inputs.sequence_ids(i)) == 1)[0]             # np.where는 numpy array와 조건을 입력받아 array 내에서 입력받은 조건과 부합하는 원소의 index를 반환합니다.
                                                                                                # 여기서 model_inputs.sequences_ids(i)는 i번째 문장에서 서로 다른 문장을 0과 1로 구분해줍니다. (직접 찢어보시면 None이 나올텐데 이건 special token이라는 의미이니 무시하시면 됩니다.)
                                                                                                # 저희는 context 문장에서 정답 단어가 토큰 기준으로 어느 위치에 있는지 찾아내는게 목표이기 때문에 context 문장에만 관심가지면 됩니다.
                                                                                                # 따라서 위 where와 sequences_ids로 context 문장이 몇 번째 토큰부터 시작하고 몇 번째 토큰에서 끝나는지 알아냅니다.
                                                                                                # 여기서 몇 번째 토큰으로 시작하고 끝나는지 알아내야 하는 이유는 offset과 연결지어 바라봐야하기 때문입니다.
            context_start = context_info.min()                                                  # context가 시작되는 토큰의 인덱스를 확인합니다.
            context_end = context_info.max()                                                    # context가 끝나는 토큰의 인덱스를 확인합니다.

            start_position = context_start                                                      # 지금 문장에서 정답이 시작되는 토큰의 인덱스를 start_position에 저장합니다.
                                                                                                # 이제 context의 시작부터 끝 토큰까지 순차적으로 돌면서 정답이 시작되는 토큰의 위치를 찾아옵니다.
                                                                                                # 여기서 우리는 offset의 각 원소를 이루는 tuple 하나가 token과 대응하고,
                                                                                                # tuple의 첫 번째 원소는 해당 token이 문자열 기준 몇 번째 문자로 시작하는 건지랑
                                                                                                # tuple의 두 번째 원소는 해당 token이 문자열 기준 몇 번째 문자로 끝나는 건지를 알고 있습니다.
                                                                                                # 결국 offset 기준으로 순회하는게 context 기준으로 순회하는거랑 마찬가지이고,
                                                                                                # 문자열 기준의 시작 위치는 offset의 원소들이랑 비교하면 확인이 될테니 이걸 기준으로 for loop를 돕니다.

            flag = False

            for p in range(context_end - context_start) :                                       # 위에서 말씀드린 loop입니다.
                start_position += 1
                if offset[start_position][0] > inputs["answer_start"][i] :                      # 만일 지금 바라보고 있는 offset의 원소로 존재하는 tuple의 첫 번째 값이(해당 토큰이 문자열 기준으로 몇 번째 문자로 시작하는지)
                                                                                                # 문자열 기준 정답의 시작 원소보다 크다면 지금 보고 있는 토큰이 정답 시작 토큰이 될 것입니다.
                                                                                                # 따라서 for loop를 멈추고 지금 보고 있는 애를 리스트에 저장합니다. 다만 첫 시작부터 1을 더하고 시작했기 때문에 넣기 전에 하나를 빼줍니다.
                    break
            else :
                flag = True
            start_positions.append(start_position - 1)

            end_position = context_end

            for p in range(context_end - context_start) :                                       # 마찬가지로 돌립니다. 차이점은 시작 위치는 앞에서부터 보니 하나씩 더해주지만 끝 위치는 하나씩 빼면서 봅니다.
                end_position -= 1
                if offset[end_position][1] < inputs["answer_end"][i] :
                    break
            else :
                flag = True
            end_positions.append(end_position + 2)

            if flag :
                start_positions[-1] = 0                                                         # 시작 위치와 끝 위치는 모두 for-else문입니다. 루프를 다 돌았는데도 갈피를 못잡는다는거는 토큰화 과정에서 본문이 잘렸기 때문에
                                                                                                # 지금은 본문의 내용으로부터 정답을 찾을 수 없다는 의미입니다. 따라서 시작과 끝을 모두 0으로 통일합니다.
                end_positions[-1] = 0

        model_inputs["start_positions"] = start_positions                                       # 이렇게 문장들에 대한 시작과 끝 위치를 다 찾아왔으니 이걸 최종적으로 반환할 딕셔너리에 넣고 끝냅니다.
        model_inputs["end_positions"] = end_positions
    return model_inputs

def get_dataset(inputs, tokenizer, collator, batch_size, training) :
    """
    데이터를 받아 torch dataset으로 load하는 함수입니다.

    arguments
        inputs : pandas DataFrame, 모델에 올릴 데이터입니다.
                 column으로 question, context를 가지고 있어야 합니다.
                 만일 training이 목적이라면 start_positions와 end_positions, offset_mapping, answer 또한 있어야 합니다.
        tokenizer : 사용할 사전학습 모델의 tokenizer입니다.
        collator : transformers collator를 받습니다.
        batch_size : 학습 / 추론에 사용할 배치 크기입니다.
        training : 해당 데이터가 학습용이면 True, 추론용이면 False를 받습니다.
    
    return
        torch DataLoader
    """
    inputs = datasets.Dataset.from_pandas(inputs)
    tokenized_inputs = inputs.map(__tokenizing,
                                  batched = True,
                                  fn_kwargs = {"training" : training,
                                               "tokenizer" : tokenizer})
    
    if training :
        columns = tokenizer.model_input_names + ["start_positions", "end_positions", "golden_answer"]
    else :
        columns = tokenizer.model_input_names

    tokenized_inputs.set_format("torch", columns = columns)
    train_dataset = torch.utils.data.DataLoader(tokenized_inputs,
                                                batch_size = batch_size,
                                                shuffle = training,
                                                collate_fn = collator)
    return train_dataset

def inference(start_logits, end_logits, context, max_len, tokenizer, tagger) :
    """
    model의 output을 참고해 추론한 결과를 반환합니다.
    해당 함수는 batch가 아닌 하나의 instance에 대해 동작합니다.
    추론 순서는 다음과 같습니다.
    1. [question + sep_token + context]로 이루어진 parameter context에서 context index 추출
    2. mask matrix 생성 - mask matrix는 위에서 추론과 관계없는 토큰을 가립니다. 즉, question과 sep_token의 내용을 가립니다.
    3. start_logit과 end_logit에서 masking 대상의 token의 logit을 아주 낮은 값으로 대체합니다.
    4. start logit과 end logit에 softmax를 적용해 probability로 해석합니다.
    5. start probability와 end probability를 담고 있는 두 vector를 outer product합니다. -> context의 길이만큼의 크기를 가진 square matrix가 생성됩니다.
       outer product로 얻은 square matrix의 row는 start, column은 end를 뜻합니다. 따라서 해당 matrix의 각 원소는 row index로 시작해 column index로 끝났을 때의 일종의 점수로 해석이 가능합니다.
       maximum likelihood estimation으로 얻은 logit을 모든 조합에 대해 표현하려면 합을 해야 하지만 outer product라는 좋은 연산이 있기 때문에 합 대신 곱으로 표현할 방법이 필요했습니다.
       간단하게 softmax로 log를 취해 곱으로 설명 가능한 값으로 만들고 score matrix를 생성합니다.
    6. 위 score matrix의 lower triangle은 설명할 수 없는 답안이 됩니다. (늦은 index로 시작해서 빠른 index로 답이 끝나는 것과 같음)
       따라서 lower triangle을 0으로 바꿔 upper triangle만 취합니다.
       ex) 실제 원소는 확률 형태로 들어가있습니다. 예시일 뿐이니 참고 바랍니다.

       3 2 1 3 5              3 2 1 3 5 
       2 1 3 4 2              0 1 3 4 2
       1 3 4 2 5      ->      0 0 4 2 5
       2 2 3 1 4              0 0 0 1 4
       5 6 8 1 2              0 0 0 0 2

    7. score matrix에서 main diagonal로부터 멀어질수록 답안의 길이가 길어집니다.
       max_len으로 제한된 답안의 길이를 얻기 위해 main diagonal로부터 적절히 떨어진 위치의 value들을 0으로 바꿉니다.
       ex) max_len = 2
       3 2 1 3 5              3 2 1 0 0
       0 1 3 4 2              0 1 3 4 0
       0 0 4 2 5      ->      0 0 4 2 5
       0 0 0 1 4              0 0 0 1 4
       0 0 0 0 2              0 0 0 0 2
    
    8. masking이 완료된 score matrix로부터 최대 값을 갖는 원소의 행과 열 index를 context에 indexing에 사용해 정답을 추출합니다.
    9. 추출한 정답을 decoding하고 적절한 post processing을 거쳐 추론된 답과 score matrix의 최대 값을 반환합니다.

    arguments
        start_logits : model이 예측한 토큰 별 답안 시작 logit
        end_logits : model이 예측 한 토큰 별 답안 종료 logit
        context : tokenizing된 context. 모델의 batch input을 받습니다.
                  실제로 batch input은 question과 context가 sep token을 기준으로 concatenated 되어있지만 편의상 context로 명명했습니다.
                  내부에서 sep token을 기준으로 context 내용을 추출해 사용합니다.
        max_len : inference된 답안의 최대 길이
        tokenizer : pretrained model의 tokenizer
        tagger : konlpy에서 제공하는 tagger (Mecab을 추천드립니다.)
    
    return
        decoded : 추론된 답안 (string)
        score : 현재 추론된 답안의 score
    """
    sep_idx = torch.where(context == tokenizer.sep_token_id)[0][0]

    mask = np.array([False] * len(context))
    mask[:sep_idx + 1] = True

    if context[sep_idx + 1] == tokenizer.sep_token_id :
        mask[sep_idx + 2] = True
    mask = torch.tensor(mask)

    start_logits[mask] = -10000
    end_logits[mask] = -10000

    start_probabilities = torch.unsqueeze(start_logits.softmax(-1), -1)
    end_probabilities = torch.unsqueeze(end_logits.softmax(-1), 0)
    scores = start_probabilities * end_probabilities
    scores = torch.triu(scores)

    max_len_mask = np.ones(shape = [scores.shape[0], scores.shape[1]]).astype(bool)
    max_len_mask = torch.tensor(max_len_mask)
    max_len_mask = torch.triu(max_len_mask, diagonal = max_len)

    scores[max_len_mask] = 0

    max_index = scores.argmax().item()
    start_index = max_index // scores.shape[1]
    end_index = max_index % scores.shape[1]
    decoded = tokenizer.decode(context[start_index:end_index], skip_special_tokens = True).replace("#", '')
    # 후처리 관련 코드는 아래부터 작성해주세요! (ex. 형태소 분석, 띄어쓰기 분석 등)
    # decoded = tagger.pos(decoded)
    # decoded =  ' '.join([pos[0] for pos in decoded if pos[1] in __TAGSETS_TO_USE])
    # decoded = spell_checker.check(decoded).checked

    decoded = remove_postposition(decoded, tagger)
    
    return decoded, scores[start_index, end_index]

def levenshtein_distance(p_start, p_end, g_answer, context, tokenizer, tagger, threshold = .5, max_len = 20) :
    """
    편집거리를 계산해 반환합니다.
    입력은 batch 단위로 들어오기 때문에 해당 batch의 평균 편집 거리를 계산해 반환합니다.

    argument
        p_start : model이 예측한 start logits
        p_end : model이 예측한 end logits
        g_answer : 답안
        context : 학습 / 추론에 사용한 batch input
        tokenizer : pretrained model에 사용된 tokenizer
        tagger : koNLPy tagger (Mecab 추천)
        threshold : inference 결과 score가 해당 parameter보다 낮을 경우 정답을 empty string으로 바꿉니다.
        max_len : inference된 답의 최대 길이
    return : batch별 평균 편집 거리
    """
    lev_dist = []
    
    for i, v in enumerate(context) : 
        true = tokenizer.decode(g_answer[i], skip_special_tokens = True)
        pred, pred_score = inference(p_start[i], p_end[i], v, max_len, tokenizer, tagger)
        if pred_score < threshold:
            pred = ''
        lev_dist.append(nltk.edit_distance(true, pred))

    return sum(lev_dist) / len(lev_dist)

def extract_accuracy (p_start, p_end, t_start, t_end) :
    """
    시작 index와 끝 index의 정확도를 계산해 반환합니다.
    batch 별로 계산됩니다.

    arguments
        p_start : model이 예측한 시작 logit
        p_end : model이 예측한 끝 logit
        t_start : true start index
        t_end : true end index
    return : list(시작 정확도, 끝 정확도)
    """
    p_start = p_start.argmax(1)
    p_end = p_end.argmax(1)
    return sum(p_start == t_start) / len(p_start), sum(p_end == t_end) / len(p_end)

def weighted_loss_fn(start_logits, end_logits, start_positions, end_positions, start_weight = .5, end_weight = .5) :
    """
    start loss와 end loss에 weighted average를 취합니다.
    모델이 end index를 잘 못맞추는 경향이 있으니 end logit에 weight를 더 주는 방향이 좋을 것입니다.
    start_weight와 end_weight의 합은 1이어야 합니다.

    arguments 
        start_logits : model이 예측한 시작 logit
        end_logit : model이 예측한 끝 logit
        start_positions : true start index
        end_positions : true end index
        start_weights : start loss의 가중치
        end_weights : end loss의 가중치
    """
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)

    ignored_index = start_logits.size(1)
    start_positions = start_positions.clamp(0, ignored_index)
    end_positions = end_positions.clamp(0, ignored_index)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index, label_smoothing = .1)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss * start_weight + end_loss * end_weight)
    return total_loss

def cleaning_output(model, data, n, collator, tagger, tokenizer, max_len, device, training = True, show_decode_result = True):   
    inferenced = []
    use_data = data.iloc[:n, :]

    if n == -1 :
        use_data = data.copy()

    inference_data = get_dataset(use_data, tokenizer, collator, 16, False)

    # n개의 출력물을 보기 위함(여기서는 index 기준 n개의 데이터 출력을 확인합니다)
    for idx, sample in enumerate(inference_data):
        sample = {k : v.to(device) for k, v in sample.items()}
        with torch.no_grad():
            with torch.cuda.amp.autocast() :
                output = model(**sample) # pre-trained Model에서 start_logits과 end_lodits을 tuple 형태로 출력
                start_logits = output["start_logits"]
                end_logits = output["end_logits"]
    
        batch_inferenced = []

        for i, v in enumerate(sample["input_ids"]) : 
            decoded, _ = inference(start_logits[i], end_logits[i], v, max_len, tokenizer).replace("#", '')
            batch_inferenced.append(decoded)

            if show_decode_result :
                print(f'------{i + (idx * len(sample["input_ids"]))}------')
                print('Context:', use_data["context"].iloc[i + (idx * len(sample["input_ids"]))])
                print('Question:', use_data["question"].iloc[i + (idx * len(sample["input_ids"]))])
                print('\n**model이 예측한 index를 활용한 정답**')
                print('Answer:', decoded)
                if training :
                    print('\n**실제 정답**')
                    print('Answer:', use_data['answer'].iloc[i + (idx * len(sample["input_ids"]))])

        # 여기서 konlpy를 활용해 예측 출력 값을 정제합니다. 
        # https://datascienceschool.net/03%20machine%20learning/03.01.02%20KoNLPy%20%ED%95%9C%EA%B5%AD%EC%96%B4%20%EC%B2%98%EB%A6%AC%20%ED%8C%A8%ED%82%A4%EC%A7%80.html
        # 위 링크 하단에 각 형태소 별 품사 태그에 대한 정보가 있습니다.
        
        # ko_sent = tagger.pos(sample['context'][start:end])
        # ko_sent = [pos[0] for pos in ko_sent if pos[1] in __TAGSETS_TO_USE] #명사군, 한자, 외국어, 숫자만 정답으로 사용
        # print('\n**model이 예측한 index를 활용한 정답**')
        # for infer in batch_inferenced :
        #     print('Answer:', infer)
        
        # if training :
        #     start = sample['offset_mapping'][sample['start_positions']][0] # train data의 실제 정답 위치를 출력 (거의 유사)
        #     end = sample['offset_mapping'][sample['end_positions']][1]
        #     print('\n**실제 정답**')
        #     print('Answer:', sample['context'][start:end])

        inferenced += batch_inferenced
    return inferenced

def remove_postposition(text, tagger) :
    """
    후처리를 진행합니다.
    추론 결과에서 조사와 기타 필요없는 문자를 삭제합니다.

    arguments
        text : 모델이 추론한 결과
        tagger : koNLPy tagger (Mecab 추천)
    return : post processing이 완료된 추론 결과
    """
    if not text :
        return text
    if ' ' in text :
        splitted = text.split()
        remains = ' '.join(splitted[:-1])
        target = splitted[-1]
    else :
        remains = ''
        target = text
    tagged = tagger.pos(target)
    use_final = True
    tag = tagged[-1]
    if '+' in tag[1] :
        ts = tag[1].split('+')
        for t in ts :
            if t not in __TAGSETS_TO_USE :
                use_final = False
    else :
        if tag[1] not in __TAGSETS_TO_USE :
            use_final = False
    
    result = remains + ' ' + ''.join([j[0] for j in tagged]) if use_final else remains + ' ' + ''.join([j[0] for j in tagged[:-1]])
    return result.strip()
