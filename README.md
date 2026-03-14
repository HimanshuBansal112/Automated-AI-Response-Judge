# Automated-AI-Response-Judge
This project automatically compares responses from two AI models, predicting which is more preferable by Human based on fact accuracy and style of writing.
It can take multiple prompt-response pairs (prompt shared between two AI), analyzing even long conversation (where 1 shared prompt + 1 response of one AI = 512 tokens per pair) and provides combined result prediction for fact and style.

## How to use?
1. Provide **a prompt** with **response from two AIs** as a pair.
2. Add another pair if needed and repeat step 1.
3. The AI analyzes all pairs and give total prediction for fact accuracy and style preference by Human.

## Output format:
It gives probability of ```[A better, B better, Tie]``` for both facts and style score.

## What it uses?
1. Uses a **supervised fine-tuned encoder** specifically trained for **human style preference of AI responses**.
2. Uses **pre-trained decoder** to evaluate **factual accuracy**.

## How to use?
1. Install required dependencies.
2. Run FastAPI server at port 8080 (it will load the trained AI models, requires 3+ GB storage).
3. Start Website to use the project.
4. Ignore training code.

## How to train?
1. Make sure that data is in csv format.
2. Run training script:
    ```bash
    python train.py --data_path path_to_csv.csv --num_epochs 3
    ```
  - Replace path_to_csv.csv with the path to your dataset.
  - Replace 3 with the number of epochs you want to train for.
3. After training, both the best model and the latest model will be saved as .pt in working directory
