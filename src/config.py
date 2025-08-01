PROMPT_DICT = {
    "fin": {
        "system": "You are a financial analyzer, given a section of a company's annual report, please answer the question according to the report context. Let's do this step by step. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> must be simple and short (e.g. just an accurate numerical value or phrases). ",
        "user_1": "### Context: ...\n ## Question: What is the average price of the products?\n ### Response:",
        "assistant_1": "There are 8 products with a total price value of 1000, so the average value is 125.00 .\n The answer is: 125.00",
    },
    "tat": {
        "system": "You are a financial analyzer, given a section of a company's annual report, please answer the question according to the report context. Let's do this step by step. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> must be simple and short (e.g. just an accurate numerical value or phrases). ",
        "user_1": "### Context: ...\n ## Question: What is the average price of the products?\n ### Response:",
        "assistant_1": "There are 8 products with a total price value of 1000, so the average value is 125.00 .\n The answer is: 125.00",
    },
    "paper": {
        "system": "You are a scientific researcher, given a section of an academic paper, please answer the question according to the context of the paper. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> should be concise with no explanation.",
        "user_1": "### Context: ...\n ## Question: Which Indian languages do they experiment with\n ### Response:",
        "assistant_1": "The answer is: Hindi, English, Kannada, Telugu, Assamese, Bengali and Malayalam",
    },
    "feta": {
        "system": "Given a section of a document, plese answer the question according to the context. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> should be a natural sentence.",
        "user_1": "### Context: ...\n ## Question: When and in what play did Platt appear at the Music Box Theatre?\n ### Response:",
        "assistant_1": "The answer is: In 2016 and 2017, Platt played in Dear Evan Hansen on Broadway at the Music Box Theatre.",
    },
    "nq": {
        "system": "Given a section of a document, plese answer the question according to the context. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> should be a paragraph from the context or a summarized short phrase.",
        "user_1": "### Context: ...\n ## Question: When will tour de france teams be announced?\n ### Response:",
        "assistant_1": "The answer is: 6 January 2018",
    },
}