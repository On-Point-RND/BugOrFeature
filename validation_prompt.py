SAMPLE_SPECT = [
                {
                    "name": "shakespeare",
                    "prompt": (
                        "ACT I. SCENE I. Verona. A public place.\n"
                        "Romeo: "
                    ),
                    "gen": {"max_new_tokens": 80, "temperature": 0.9, "top_k": 50},
                },
                {
                    "name": "hello_world",
                    "prompt": "Hello, world! ",
                    "gen": {"max_new_tokens": 64, "temperature": 0.8, "top_k": 40},
                },
                {
                    "name": "python_code",
                    "prompt": (
                        "Write a Python function to compute the first 20 Fibonacci numbers:\n"
                        "def fib(n):\n    "
                    ),
                    "gen": {"max_new_tokens": 96, "temperature": 0.95, "top_k": 50},
                },
            ]