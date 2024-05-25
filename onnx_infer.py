import time
import os
import numpy as np
import onnxruntime as ort
from src.rwkv_tokenizer import RWKV_TOKENIZER  # 切换到速度更快的分词器
from src.sampler import sample_logits_numpy as sample_logits

if __name__ == '__main__':
    # Load the ONNX model
    print("Loading model and tokenizer...")
    model_path = './onnx/quant/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.sim.avx2.onnx/RWKV-x060-World-1B6-v2.onnx'
    session = ort.InferenceSession(model_path)

    # Load the tokenizer
    tokenizer = RWKV_TOKENIZER("./asset/rwkv_vocab_v20230424.txt")
    print("Done.")

    # Set the initial string and parameters for inference
    initial_string = "Elon Musk has"
    batch_size = 3
    TEMPERATURE = 1
    TOP_P = 0
    LENGTH_PER_TRIAL = 100

    # Encode the initial string
    encoded_input = tokenizer.encode([initial_string] * batch_size)
    token = np.array(encoded_input).astype(np.int64).transpose()

    # Initialize the state
    state = np.zeros((batch_size, session.get_inputs()[1].shape[1], session.get_inputs()[1].shape[2]), dtype=np.float32)
    # print(token, token.shape)
    print("Prefill the state...")
    # Prefill the state by running the initial tokens through the model
    for t in token:
        ort_inputs = {'token': t, 'input_state': state}
        ort_outs = session.run(None, ort_inputs)
        out, state = ort_outs
    print("Done.")

    # Reset token to only contain the initial encoded input
    token = token.transpose()
    # Start timing
    start_time = time.time()

    # Inference loop
    for step in range(LENGTH_PER_TRIAL):
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P)
        token = np.concatenate((token, np.expand_dims(token_sampled, axis=1)), axis=1)
        # Run the model (inference)
        ort_inputs = {'token': token_sampled, 'input_state': state}
        ort_outs = session.run(None, ort_inputs)
        # Sample logits
        out, state = ort_outs

        # Clear the screen and print the results
        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token.tolist())
        for i, seq in enumerate(decoded_sequences):
            print(f"Batch {i + 1}: {seq}")

    # End timing
    end_time = time.time()

    # Calculate and print generation speed
    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")