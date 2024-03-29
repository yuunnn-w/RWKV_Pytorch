from src.old_tokenizer import RWKV_TOKENIZER
from src.rwkv_tokenizer import RWKV_TOKENIZER as TRIE_TOKENIZER
from icecream import ic
import requests

def timed(f):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        print(f"{f.__name__} took {end-start} seconds")
        return res
    return wrapper


tokenizer1 = RWKV_TOKENIZER("./asset/rwkv_vocab_v20230424.txt")
tokenizer2 = TRIE_TOKENIZER("./asset/rwkv_vocab_v20230424.txt")


# download the tiny shakespear dataset
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(data_url).text

def test_equal():
    text = "hello world and 一些中文 to make sure the tokenizers are the same, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, @, #, $, %, ^, &, *, (, ), _, +, =, -, [, ], {, }, |, \\, ;, :, ', \", ,, <, >, ., /, ?, !, ~, `"
    result1 = tokenizer1.encode([text])
    result2 = tokenizer2.encode([text])
    assert tokenizer1.encode([text] * 4) == tokenizer2.encode([text] * 4)

@timed
def test_tokenizer1():
    tokenizer1.encode([text])

@timed
def test_tokenizer2():
    tokenizer2.encode([text])

if __name__ == '__main__':
    
    test_equal()
    test_tokenizer1()
    test_tokenizer2()