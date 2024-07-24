from datasets import load_dataset

#dataset = load_dataset("lambada", split="test")

dataset = load_dataset("wikitext", 'wikitext-2-v1', split="test")
dataset.save_to_disk("datasets/wiki2")



