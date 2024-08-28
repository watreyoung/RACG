### Transform the original json file into DataLoader format.

```bash
./code_generation/preprocess.ipynb
```

### Get embedding of the Dataset including train, test and dev

```bash
./code_generation/inference_embedding.sh
```

### Get the index

```bash
./test.sh
```

### Create the retrieval-augmented file according to the index

```bash
./code_generation/results/get_query_passage.ipynb
```

