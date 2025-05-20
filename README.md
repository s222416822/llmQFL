
#Installations:

- pip install qiskit
- pip install qiskit_machine_learning
- pip install qiskit_algorithms
- pip install pylatexenc
- pip install --quiet transformers accelerate evaluate datasets peft bitsandbytes

#Files and Folders
There are 4 folders:
1. ibm_quantum_computer: For running on Quantum Computer
2. qfl_llm_qcnn_deepseek_tweet: For running for Deepseek LLM with QFL on TweetEval-Sentiment Dataset
3. qfl_llm_qcnn_gpt2_tweet: For GPT2, TWEETEVAL
4. qfl_llm_vqc_llama_genomic: VQC with LLaMA; Genomic Dataset


#Running code:
- python main.py (In their respective folders) 



#References:
1. https://qiskit-community.github.io/qiskit-machine-learning/
2. https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html https://www.cs.toronto.edu/~kriz/cifar.html
2. https://drlee.io/fine-tuning-gpt-2-for-sentiment-analysis-94ebdd7b5b24
3. https://medium.com/@oshananoah/from-tweets-to-insights-fine-tuning-gpt-for-sentiment-analysis-1950fe958778
4. https://blog.devgenius.io/sculpting-language-gpt-2-fine-tuning-with-lora-1caf3bfbc3c6
5. https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora#:~:text=QLoRA%20and%20LoRA%20both%20are,of%20a%20standalone%20finetuning%20technique.
6. https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments.load_best_model_at_end
7. https://dataman-ai.medium.com/fine-tune-a-gpt-lora-e9b72ad4ad3
11. https://huggingface.co/docs/peft/en/package_reference/lora
12. https://github.com/microsoft/LoRA
13. https://github.com/huggingface/peft
14. https://huggingface.co/blog/peft
15. https://www.datacamp.com/tutorial/fine-tuning-large-language-models
16. https://huggingface.co/docs/transformers/en/index
17. https://huggingface.co/docs/transformers/en/model_doc/gpt2
18. https://huggingface.co/meta-llama/Llama-3.2-1B
19. https://huggingface.co/deepseek-ai/deepseek-llm-7b-base


Note* Results obtained by running the same codes in Notebook in Google Colab Pro+. This repo optimized/formatted for submission and ease use.

