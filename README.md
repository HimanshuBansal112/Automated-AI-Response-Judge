# Automated-AI-Response-Judge
This project automatically compares responses from two AI models, predicting which is more preferable by Human based on fact accuracy and style of writing.
It can take multiple prompt-response pairs (prompt shared between two AI), analyzing even long conversation (where 1 shared prompt + 1 response of one AI = 512 tokens per pair) and provides combined result prediction for fact and style.

# How to use?
1. Provide **a prompt** with **response from two AIs** as a pair.
2. Add another pair if needed and repeat step 1.
3. The AI analyzes all pairs and give total prediction for fact accuracy and style preference by Human.

# Output format:
It gives probability of ```[A better, B better, Tie]``` for both facts and style score.

# What it uses?
1. Uses a **supervised fine-tuned encoder** specifically trained for **human style preference of AI responses**.
2. Uses **pre-trained decoder** to evaluate **factual accuracy**.

# How to use?
1. Install required dependencies.
2. Run FastAPI server at port 8080 (it will load the trained AI models, requires 3+ GB storage).
3. Start Website to use the project.
