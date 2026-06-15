import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)

class VocabVisualizer:
    """Visualizes tokenizer vocabulary sizes using Treemaps or other charts."""

    # Common model vocab sizes (can be extended or loaded dynamically)
    DEFAULT_MODELS = [
        {"model": "Qwen2.5/Qwen-VL", "vocab": 151936, "family": "Qwen"},
        {"model": "Llama-3", "vocab": 128256, "family": "Llama"},
        {"model": "GPT-4o", "vocab": 200000, "family": "OpenAI"},
        {"model": "DeepSeek-V3", "vocab": 129280, "family": "DeepSeek"},
        {"model": "Mistral-Large", "vocab": 32768, "family": "Mistral"},
        {"model": "Gemma-2", "vocab": 256000, "family": "Google"},
        {"model": "GPT-2", "vocab": 50257, "family": "OpenAI"},
        {"model": "BERT", "vocab": 30522, "family": "Google"},
        {"model": "T5", "vocab": 32128, "family": "Google"},
        {"model": "Yi-34B", "vocab": 64000, "family": "01.AI"},
        {"model": "ChatGLM3", "vocab": 65024, "family": "GLM"},
        {"model": "InternLM2.5", "vocab": 92544, "family": "InternLM"},
        {"model": "Baichuan2", "vocab": 125696, "family": "Baichuan"},
        {"model": "MiniMax-M2.5", "vocab": 200064, "family": "MiniMax"},
        {"model": "ERNIE-4.0", "vocab": 103424, "family": "Baidu"} # Approx
    ]

    def __init__(self, models: Optional[List[Dict]] = None):
        self.models = models if models else self.DEFAULT_MODELS.copy()

    def add_model_from_id(self, model_id: str, family: str = "Custom", trust_remote_code: bool = False) -> None:
        """Loads a tokenizer from HF model ID and adds it to the list."""
        try:
            from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer
            logger.info("Loading tokenizer for %s...", model_id)
            tokenizer = hf_load_tokenizer(
                model_id,
                security={
                    "trust_remote_code": trust_remote_code,
                    "allow_network": True,
                    "local_files_only": False,
                },
            )
            vocab_size = len(tokenizer) # or tokenizer.vocab_size

            # Check if exists, update if so
            for m in self.models:
                if m["model"] == model_id:
                    m["vocab"] = vocab_size
                    return

            self.models.append({
                "model": model_id,
                "vocab": vocab_size,
                "family": family
            })
            logger.info("Added %s: %d tokens", model_id, vocab_size)
        except Exception as e:
            logger.warning("Failed to load tokenizer for %s: %s", model_id, e)

    def generate_treemap(self, output_path: str = "vocab_treemap.html") -> Any:
        """Generates a Treemap where area represents vocab size."""
        df = pd.DataFrame(self.models)

        # Add a formatted label
        df["label"] = df["model"] + "<br>" + df["vocab"].apply(lambda x: f"{x:,}")

        fig = px.treemap(
            df,
            path=[px.Constant("All Models"), 'family', 'model'],
            values='vocab',
            color='vocab',
            color_continuous_scale='Viridis',
            hover_data=['vocab'],
            title="LLM Vocabulary Size Comparison (Treemap)"
        )

        fig.update_traces(textinfo="label+value")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        fig.write_html(output_path)
        return output_path

    def generate_bar_chart(self, output_path: str = "vocab_bar.html") -> Any:
        """Generates a Bar Chart comparing vocab sizes."""
        df = pd.DataFrame(self.models).sort_values("vocab", ascending=True)

        fig = px.bar(
            df,
            x="vocab",
            y="model",
            color="family",
            orientation='h',
            text="vocab",
            title="LLM Vocabulary Size Ranking"
        )

        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=800)

        fig.write_html(output_path)
        return output_path

    def generate_single_distribution(self, model_id: str, output_path: str = "vocab_dist.html", plot_type: str = "treemap", trust_remote_code: bool = False) -> Optional[Any]:
        """Generates a visualization of token types within a single tokenizer."""
        try:
            import unicodedata

            from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

            logger.info("Analyzing tokenizer distribution for %s...", model_id)
            tokenizer = hf_load_tokenizer(
                model_id,
                security={
                    "trust_remote_code": trust_remote_code,
                    "allow_network": True,
                    "local_files_only": False,
                },
            )
            vocab = tokenizer.get_vocab()

            # Data structure
            data = []

            # Compression analysis sample texts
            samples = {
                "English": "The quick brown fox jumps over the lazy dog. Machine learning is fascinating and powerful.",
                "Chinese": (
                    "Cishi cike, women zhengzai jinxing ziran yuyan chuli de yanjiu. "
                    "Shendu xuexi gaibianle shijie, rang jiqi neng lijie renlei yuyan."
                ),
                "Code": "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
                "Math": "The area of a circle is calculated as A = pi * r^2. E = mc^2 is famous."
            }

            # Populate data for visualization
            for token, idx in vocab.items():
                category = "Other/Symbol"
                subcategory = "Misc"

                # Clean token (remove Ġ,   etc.)
                clean_token = token.replace('Ġ', '').replace(' ', '').replace('##', '').replace(' ', '')
                token_len = len(clean_token)

                if idx in tokenizer.all_special_ids:
                    category = "Special"
                    subcategory = "Control"
                else:
                    if not clean_token:
                        category = "Special"
                        subcategory = "Empty/Format"
                    elif all(c.isascii() and c.isalpha() for c in clean_token):
                        category = "English/Latin"
                        if token_len <= 3:
                            subcategory = "Short (<4)"
                        elif token_len <= 6:
                            subcategory = "Medium (4-6)"
                        else:
                            subcategory = "Long (>6)"
                    elif all(c.isdigit() for c in clean_token):
                        category = "Digits"
                        subcategory = "Pure Digits"
                    elif any('\u4e00' <= c <= '\u9fff' for c in clean_token):
                        category = "Chinese"
                        if token_len == 1:
                            subcategory = "Single Char"
                        elif token_len == 2:
                            subcategory = "Bigram"
                        else:
                            subcategory = "Phrase"
                    elif any('\u0400' <= c <= '\u04FF' for c in clean_token):
                        category = "Cyrillic"
                        subcategory = "General"
                    else:
                        category = "Other/Symbol"
                        if any(c.isascii() for c in clean_token):
                            subcategory = "ASCII Punctuation"
                        else:
                            subcategory = "Unicode Symbol"

                data.append({
                    "Token": token,
                    "Category": category,
                    "Subcategory": subcategory,
                    "Length": token_len,
                    "FirstChar": clean_token[0] if clean_token else "",
                    "Count": 1
                })

            df = pd.DataFrame(data)

            if plot_type == "compression-radar":
                # Compression Efficiency Analysis
                radar_data = []
                for domain, text in samples.items():
                    tokens = tokenizer.tokenize(text)
                    num_tokens = len(tokens)
                    num_chars = len(text)
                    # Characters per Token (Higher is better efficiency)
                    cpt = num_chars / num_tokens if num_tokens > 0 else 0

                    radar_data.append(dict(
                        Domain=domain,
                        CPT=cpt,
                        Model=model_id
                    ))

                df = pd.DataFrame(radar_data)
                fig = px.line_polar(
                    df, r='CPT', theta='Domain', line_close=True,
                    title=f"Token Compression Efficiency (Chars/Token): {model_id}",
                    markers=True
                )
                fig.update_traces(fill='toself')
                fig.write_html(output_path)
                return output_path

            elif plot_type == "unicode-sunburst":
                # Detailed Unicode Block Analysis
                block_counts = {}

                for token, _idx in vocab.items():
                    # Clean token
                    clean_token = token.replace('Ġ', '').replace(' ', '').replace('##', '').replace(' ', '')
                    if not clean_token:
                        continue

                    # Analyze first character's block
                    try:
                        char = clean_token[0]
                        # Simple block mapping (could be more exhaustive)
                        code = ord(char)
                        block = "Unknown"

                        if 0x0000 <= code <= 0x007F:
                            block = "Basic Latin"
                        elif 0x0080 <= code <= 0x00FF:
                            block = "Latin-1 Supplement"
                        elif 0x4E00 <= code <= 0x9FFF:
                            block = "CJK Unified Ideographs"
                        elif 0x0400 <= code <= 0x04FF:
                            block = "Cyrillic"
                        elif 0x0600 <= code <= 0x06FF:
                            block = "Arabic"
                        elif 0x3040 <= code <= 0x309F:
                            block = "Hiragana"
                        elif 0x1F600 <= code <= 0x1F64F:
                            block = "Emoticons"
                        else:
                            # Fallback using unicodedata name
                            try:
                                name = unicodedata.name(char)
                                block = name.split()[0]  # Rough grouping
                            except (ValueError, KeyError):
                                block = "Other"

                        block_counts[block] = block_counts.get(block, 0) + 1
                    except (ValueError, KeyError):
                        pass

                df_blocks = pd.DataFrame(list(block_counts.items()), columns=["Block", "Count"])
                fig = px.sunburst(
                    df_blocks,
                    path=['Block'],
                    values='Count',
                    title=f"Unicode Block Distribution: {model_id}"
                )
                fig.write_html(output_path)
                return output_path

            elif plot_type == "vocab-map":
                # 2D Heatmap of Index Space
                # Map indices to a grid (e.g. 256 or 512 width)
                # Color by Category

                width = 512
                height = (len(vocab) + width - 1) // width

                # Create grid arrays
                import numpy as np
                grid_category = np.full((height, width), 0, dtype=int) # 0=Special/Other

                # Define numeric codes for categories
                # 0: Special/Other
                # 1: English
                # 2: Chinese
                # 3: Digits
                # 4: Cyrillic
                # 5: Symbol

                for _idx, item in enumerate(data):
                    token_idx = tokenizer.convert_tokens_to_ids(item["Token"])
                    cat = item["Category"]

                    code = 0
                    if cat == "English/Latin":
                        code = 1
                    elif cat == "Chinese":
                        code = 2
                    elif cat == "Digits":
                        code = 3
                    elif cat == "Cyrillic":
                        code = 4
                    elif cat == "Other/Symbol":
                        code = 5
                    elif cat == "Special":
                        code = 6

                    r, c = divmod(token_idx, width)
                    if r < height:
                        grid_category[r, c] = code

                # Create heatmap
                fig = px.imshow(
                    grid_category,
                    labels=dict(x="Index % 512", y="Index // 512", color="Category"),
                    color_continuous_scale=[
                        (0.00, "gray"),   # 0 Special/Other
                        (0.16, "gray"),
                        (0.16, "blue"),   # 1 English
                        (0.33, "blue"),
                        (0.33, "red"),    # 2 Chinese
                        (0.50, "red"),
                        (0.50, "green"),  # 3 Digits
                        (0.66, "green"),
                        (0.66, "orange"), # 4 Cyrillic
                        (0.83, "orange"),
                        (0.83, "purple"), # 5 Symbol
                        (1.00, "purple")
                    ],
                    title=f"Vocabulary Map (Index Space): {model_id}"
                )

                # Add custom legend via layout annotations or just description
                fig.update_layout(
                    title_subtitle_text="Gray: Other | Blue: English | Red: Chinese | Green: Digits | Orange: Cyrillic | Purple: Symbol"
                )
                fig.write_html(output_path)
                return output_path

            elif plot_type == "digit-coverage":
                # Check coverage of numbers 0-9999
                # Are they single tokens?

                years = [str(y) for y in range(1900, 2030)]
                small_ints = [str(i) for i in range(0, 1000)]

                coverage_data = []

                def check_single(num_str) -> Any:
                    ids = tokenizer.encode(num_str, add_special_tokens=False)
                    return len(ids) == 1

                # Check 0-1000
                single_count = sum(1 for x in small_ints if check_single(x))
                coverage_data.append({"Set": "Integers 0-999", "Single Token %": (single_count/1000)*100})

                # Check Years
                year_count = sum(1 for x in years if check_single(x))
                coverage_data.append({"Set": "Years 1900-2029", "Single Token %": (year_count/len(years))*100})

                fig = px.bar(
                    pd.DataFrame(coverage_data),
                    x="Set", y="Single Token %",
                    color="Set",
                    title=f"Numeric Token Coverage: {model_id}",
                    range_y=[0, 100]
                )
                fig.write_html(output_path)
                return output_path

            elif plot_type == "subword-fertility":
                # Average tokens per word analysis
                fertility_data = []
                test_words = {
                    "Common English": ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I"],
                    "Complex English": ["extraordinary", "phenomenon", "consciousness", "implementation", "development"],
                    "Chinese Common": ["de", "yi", "shi", "zai", "bu", "le", "you", "he", "ren", "zhe"],
                    "Chinese Complex": [
                        "ziran yuyan chuli",
                        "rengong zhineng",
                        "jiqi xuexi",
                        "shenjing wangluo",
                        "shendu xuexi",
                    ],
                    "Code": ["function", "return", "import", "class", "print", "def", "async", "await"],
                    "Emoji": ["😀", "👍", "🚀", "🤖", "🧠"]
                }

                for category, words in test_words.items():
                    total_tokens = 0
                    total_chars = 0
                    for w in words:
                        t = tokenizer.tokenize(w)
                        total_tokens += len(t)
                        total_chars += len(w)

                    avg_tokens = total_tokens / len(words)
                    fertility_data.append({
                        "Category": category,
                        "Avg Tokens/Word": avg_tokens,
                        "Example Count": len(words)
                    })

                fig = px.bar(
                    pd.DataFrame(fertility_data),
                    x="Category", y="Avg Tokens/Word",
                    color="Category",
                    title=f"Subword Fertility (Lower is Better): {model_id}",
                    text="Avg Tokens/Word"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.write_html(output_path)
                return output_path

            elif plot_type == "special-tokens":
                # Special Tokens Analysis
                special_tokens = []
                for t, i in tokenizer.get_added_vocab().items():
                    special_tokens.append({"Token": t, "ID": i, "Type": "Added"})

                # Also check standard special tokens
                for attr in tokenizer.special_tokens_map:
                    val = getattr(tokenizer, attr, None)
                    if val:
                        # Handle list or single string
                        vals = val if isinstance(val, list) else [val]
                        for v in vals:
                            # Avoid duplicates if possible, or just list them
                            special_tokens.append({"Token": str(v), "ID": tokenizer.convert_tokens_to_ids(v), "Type": attr})

                if not special_tokens:
                    # Fallback if no special tokens found/exposed
                    special_tokens.append({"Token": "None", "ID": 0, "Type": "None"})

                df_special = pd.DataFrame(special_tokens).drop_duplicates(subset=["Token"])

                fig = px.scatter(
                    df_special,
                    x="ID", y="Type",
                    text="Token",
                    color="Type",
                    title=f"Special Token Map: {model_id}",
                    hover_data=["ID", "Token"]
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(height=max(400, len(df_special)*20 + 200))
                fig.write_html(output_path)
                return output_path

            # ... (Existing logic for treemap/hist/first-char) ...

            if plot_type == "length-hist":
                # Histogram of token lengths
                # Clip very long tokens for visualization
                df["LengthClipped"] = df["Length"].clip(upper=20)

                fig = px.histogram(
                    df,
                    x="LengthClipped",
                    color="Category",
                    nbins=20,
                    title=f"Token Length Distribution: {model_id}",
                    labels={"LengthClipped": "Token Length (Characters)"},
                    barmode="stack"
                )
                fig.update_layout(xaxis_title="Length (clipped at 20)")

            elif plot_type == "first-char":
                # Heatmap of starting characters (ASCII only for readability)
                # Filter for ASCII first chars
                df_ascii = df[df["FirstChar"].apply(lambda x: len(x)==1 and x.isascii() and x.isprintable())]

                char_counts = df_ascii["FirstChar"].value_counts().reset_index()
                char_counts.columns = ["Char", "Count"]
                char_counts = char_counts.sort_values("Char")

                fig = px.bar(
                    char_counts,
                    x="Char",
                    y="Count",
                    title=f"Token First Character Distribution (ASCII): {model_id}"
                )

            else: # Default Treemap
                df_agg = df.groupby(["Category", "Subcategory"]).size().reset_index(name="Count")
                df_agg["Root"] = model_id

                fig = px.treemap(
                    df_agg,
                    path=['Root', 'Category', 'Subcategory'],
                    values='Count',
                    color='Count',
                    color_continuous_scale='RdBu',
                    title=f"Token Distribution: {model_id} (Vocab: {len(vocab):,})"
                )
                fig.update_traces(textinfo="label+value+percent entry")

            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            fig.write_html(output_path)
            return output_path

        except Exception as e:
            logger.warning("Failed to analyze tokenizer for %s: %s", model_id, e)
            return None
