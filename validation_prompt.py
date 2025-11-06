SAMPLE_SPECT = SAMPLE_SPECT = [
    # ──────────────── 1. Deterministic / Factual Recall (Low Temp) ────────────────
    {
        "name": "capital_france_0.0",
        "prompt": "The capital of France is ",
        "gen": {"max_new_tokens": 20, "temperature": 0.0, "top_k": 1},
    },
    {
        "name": "president_usa_0.1",
        "prompt": "The current president of the United States is ",
        "gen": {"max_new_tokens": 30, "temperature": 0.1, "top_k": 5},
    },
    {
        "name": "water_formula_0.0",
        "prompt": "The chemical formula for water is ",
        "gen": {"max_new_tokens": 10, "temperature": 0.0, "top_k": 1},
    },

    # ──────────────── 2. Simple Language Fluency (Medium Temp) ────────────────
    {
        "name": "hello_world_0.7",
        "prompt": "Hello, world! ",
        "gen": {"max_new_tokens": 64, "temperature": 0.7, "top_k": 40},
    },
    {
        "name": "weather_today",
        "prompt": "The weather today is ",
        "gen": {"max_new_tokens": 50, "temperature": 0.8, "top_k": 30},
    },
    {
        "name": "continue_sentence",
        "prompt": "She opened the door and saw ",
        "gen": {"max_new_tokens": 70, "temperature": 0.9, "top_k": 50},
    },

    # ──────────────── 3. Structured Reasoning / Math (Low-Medium Temp) ────────────────
    {
        "name": "addition_0.1",
        "prompt": "What is 123 + 456? The answer is ",
        "gen": {"max_new_tokens": 20, "temperature": 0.1, "top_k": 10},
    },
    {
        "name": "next_number",
        "prompt": "The next number in the sequence 2, 4, 8, 16 is ",
        "gen": {"max_new_tokens": 15, "temperature": 0.3, "top_k": 20},
    },
    {
        "name": "days_in_year",
        "prompt": "How many days are in a non-leap year? ",
        "gen": {"max_new_tokens": 10, "temperature": 0.0, "top_k": 1},
    },

    # ──────────────── 4. Code Generation (Higher Temp, Creative) ────────────────
    {
        "name": "python_fib_0.95",
        "prompt": (
            "Write a Python function to compute the first 20 Fibonacci numbers:\n"
            "def fib(n):\n    "
        ),
        "gen": {"max_new_tokens": 96, "temperature": 0.95, "top_k": 50},
    },
    {
        "name": "hello_func_js",
        "prompt": "Write a JavaScript function that prints 'Hello, World!':\nfunction hello() {\n  ",
        "gen": {"max_new_tokens": 60, "temperature": 0.9, "top_k": 40},
    },
    {
        "name": "joke",
        "prompt": "Tell me a joke ",
        "gen": {"max_new_tokens": 100, "temperature": 0.4, "top_k": 100},
    },

    # ──────────────── 5. Creative / Open-Ended (High Temp) ────────────────
    {
        "name": "shakespeare_style",
        "prompt": "ACT I. SCENE I. Verona. A public place.\nRomeo: ",
        "gen": {"max_new_tokens": 80, "temperature": 0.9, "top_k": 50},
    },
    {
        "name": "story_start",
        "prompt": "Once upon a time, in a forest made of glass, ",
        "gen": {"max_new_tokens": 100, "temperature": 1.0, "top_k": 60},
    },
    {
        "name": "poem_about_ai",
        "prompt": "Write a short poem about artificial intelligence:\n",
        "gen": {"max_new_tokens": 90, "temperature": 0.95, "top_k": 50},
    },

    # ──────────────── 6. Stress Test: Repetition & Coherence ────────────────
    {
        "name": "repeat_test",
        "prompt": "Repeat the word 'apple' five times: ",
        "gen": {"max_new_tokens": 30, "temperature": 0.0, "top_k": 1},
    },
    {
        "name": "long_coherence",
        "prompt": "Explain photosynthesis in simple terms: ",
        "gen": {"max_new_tokens": 120, "temperature": 0.8, "top_k": 40},
    },
]