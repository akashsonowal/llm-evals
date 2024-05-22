# to select the evaluator we have different steps, this script is not to evaluate a evaluator

def get_retrieval_score(references, generated): # reference is the ground truth chunk
    matches = np.zeros(len(references))
    for i in range(len(references)):
        reference_source = references[i]["source"].split("#")[0]
        if not reference_source:
            matches[i] = 1
            continue
        for source in generated[i]["sources"]:
            # sections don't have to perfectly match
            if reference_source == source.split("#")[0]:
                matches[i] = 1
                continue
    retrieval_score = np.mean(matches)
    return retrieval_score

evaluation_system_content = """
    Your job is to rate the quality of our generated answer {generated_answer}
    given a query {query} and a reference answer {reference_answer}.
    Your score has to be between 1 and 5.
    You must return your response in a line with only the score.
    Do not return answers in any other format.
    On a separate line provide your reasoning for the score as well.
    """

def evaluate_responses(references, ): # references are generated in the cold start synthetic data
    pass # quality score


def run_experiment():
    pass 

# experiments

# check with number of contexts

# Chunk sizes

# number of chunks

## embedding model

## change LLM


## Popular
## Tip: use this small chunks but retrieve surrounding chunks https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#decoupling-chunks-used-for-retrieval-vs-chunks-used-for-synthesis
## or use multiple embeddings per documents e.g. summary https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/multi_vector/

## 4 chars = 1 token

# we cab finetune to have longer context length also checkout ROPE scaling
## Tip try inscreasing number of chunks
