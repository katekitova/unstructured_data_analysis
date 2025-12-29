import re
from collections import Counter
from natasha import Segmenter, MorphVocab, NewsMorphTagger, NewsEmbedding, Doc
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class ColorAnalyzer:
    def __init__(self, dictionary_file="dictionary.txt"):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.embeddings = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.embeddings)
        self.stemmer = SnowballStemmer("russian")
        self.color_dict = self._load_color_dictionary_from_file(dictionary_file)
        self.auto_detected_complex_colors = set()
        self.output_dir = Path("figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    #  загрузка словаря
    def _load_color_dictionary_from_file(self, filename):
        colors = set()
        complex_colors = set()
        simple_colors = set()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    color = line.strip().lower()
                    if not color:
                        continue
                    colors.add(color)
                    if '-' in color:
                        complex_colors.add(color)
                    else:
                        simple_colors.add(color)
            print(f"Загружено {len(colors)} цветов из файла {filename}")
            print(f"Из них сложных цветов: {len(complex_colors)}")
            print(f"Простых цветов: {len(simple_colors)}")
        except FileNotFoundError:
            print(f"Файл {filename} не найден! Работа без словаря цветов невозможна.")
            return {
                'full_colors': set(),
                'color_stems': set(),
                'complex_colors': set(),
                'simple_colors': set(),
                'lemma_map': {}
            }
        except Exception as e:
            print(f"Ошибка при чтении файла {filename}: {e}")
            return {
                'full_colors': set(),
                'color_stems': set(),
                'complex_colors': set(),
                'simple_colors': set(),
                'lemma_map': {}
            }
        # основы по простым цветам
        color_stems = set()
        lemma_map = {}
        for color in simple_colors:
            st = self.stemmer.stem(color)
            color_stems.add(st)
            lemma_map.setdefault(st, color)
        return {
            'full_colors': colors,
            'color_stems': color_stems,
            'complex_colors': complex_colors,
            'simple_colors': simple_colors,
            'lemma_map': lemma_map
        }

    #  предобработка и морфология
    def preprocess_text(self, text):
        return re.sub(r'\s+', ' ', text).strip().lower()

    def morphological_analysis(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        tokens = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemma = (token.lemma or token.text).lower()
            tokens.append({
                'text': token.text.lower(),
                'lemma': lemma,
                'pos': token.pos if hasattr(token, 'pos') else 'UNKN',
                'stem': self.stemmer.stem(lemma)
            })
        return tokens

    def is_simple_color_lemma(self, s):
        return s in self.color_dict['simple_colors']

    def find_color_by_stem(self, stem):
        return self.color_dict['lemma_map'].get(stem)

    def find_simple_color_for_part(self, part):
        st = self.stemmer.stem(part.lower())
        if st in self.color_dict['color_stems']:
            base = self.color_dict['lemma_map'].get(st)
            if base and '-' not in base:
                return base
        return None


    # правило для сложных с >=2 дефисами
    def normalize_multi_hyphen_color(self, token_text, token_lemma):
        if token_text in self.color_dict['complex_colors']:
            return token_text
        parts_text = token_text.split('-')
        if len(parts_text) < 3:
            return None
        parts_lemma = token_lemma.split('-') if token_lemma else parts_text
        last_lemma = parts_lemma[-1] if parts_lemma else parts_text[-1]
        if self.is_simple_color_lemma(last_lemma):
            normalized_last = last_lemma
        else:
            normalized_last = self.find_simple_color_for_part(last_lemma)
            if not normalized_last:
                return None
        prefix = '-'.join(parts_text[:-1])
        result = f"{prefix}-{normalized_last}"
        self.auto_detected_complex_colors.add(result)
        return result

    #  извлечение цветов
    def extract_colors(self, tokens):
        colors_found = []
        if not self.color_dict['full_colors']:
            return colors_found
        for t in tokens:
            txt = t['text']
            lem = t['lemma']
            st = t['stem']
            # print(f"\nТокен {i}: '{txt}' -> лемма '{lem}'")
            # точное совпадение
            if lem in self.color_dict['full_colors']:
                colors_found.append(lem)
                # print(f" Найден по лемме: {lem}")
                continue
            if txt in self.color_dict['full_colors']:
                colors_found.append(txt)
                # print(f" Найден по тексту: {txt}")
                continue
            # простой по основе
            if st in self.color_dict['color_stems']:
                base = self.find_color_by_stem(st)
                if base:
                    colors_found.append(base)
                    # print(f" Найден по основе: {base}")
                    continue
            # дефисные варианты
            if '-' in txt:
                # print(f"  Обнаружено сложное слово: '{txt}'")
                if txt in self.color_dict['complex_colors']:
                    colors_found.append(txt)
                    # print(f" Найден сложный цвет из словаря: {txt}")
                    continue
                # правило для >=2 дефисов
                normalized = self.normalize_multi_hyphen_color(txt, lem)
                if normalized:
                    colors_found.append(normalized)
                    # print(f"Извлечен по правилу (последняя часть-лемма): {normalized}")
                    continue
                # для 1 дефиса — по хвостовой лемме
                parts = txt.split('-')
                if len(parts) == 2:
                    lemma_parts = lem.split('-') if lem else parts
                    last_lemma = lemma_parts[-1]
                    if self.is_simple_color_lemma(last_lemma):
                        candidate = f"{parts[0]}-{last_lemma}"
                    else:
                        tail = self.find_simple_color_for_part(last_lemma)
                        candidate = f"{parts[0]}-{tail}" if tail else None
                    if candidate:
                        colors_found.append(candidate)
                        self.auto_detected_complex_colors.add(candidate)
                        # print(f"Извлечен двучастный по хвостовой лемме: {candidate}")
                        continue
                # print(f"Не удалось интерпретировать как цвет по правилам")
        return colors_found

    # статистика и печать
    def calculate_statistics(self, text, colors_found, tokens):
        NON_WORD_POS = {'PUNCT', 'SPACE'}
        all_word_tokens = [
            t for t in tokens
            if (t.get('pos') not in NON_WORD_POS) and (t.get('lemma') and t['lemma'].strip())
        ]
        total_tokens_no_punct = len(all_word_tokens)
        unique_lemmas_all = set(t['lemma'] for t in all_word_tokens)
        unique_words_all = len(unique_lemmas_all)
        color_counter = Counter(colors_found)
        total_color_mentions = len(colors_found)
        complex_colors = [c for c in colors_found if '-' in c]
        if total_color_mentions > 0:
            relative_freq_to_words = {
                c: (n / total_tokens_no_punct if total_tokens_no_punct else 0.0)
                for c, n in color_counter.items()
            }
            relative_freq_to_colors = {c: n / total_color_mentions for c, n in color_counter.items()}
            max_count = max(color_counter.values()) if color_counter else 0
            most_common_colors = sorted([c for c, n in color_counter.items() if n == max_count])
            most_common_color = most_common_colors[0] if most_common_colors else None
            most_common_count = max_count
        else:
            relative_freq_to_words = {}
            relative_freq_to_colors = {}
            most_common_colors = []
            most_common_color, most_common_count = None, 0
        stats = {
            'total_words': total_tokens_no_punct,
            'unique_words': unique_words_all,
            'total_color_mentions': total_color_mentions,
            'unique_colors': len(set(colors_found)),
            'color_frequency_absolute': dict(color_counter),
            'color_frequency_relative_to_words': relative_freq_to_words,
            'color_frequency_relative_to_colors': relative_freq_to_colors,
            'most_common_color': most_common_color,
            'most_common_colors': most_common_colors,
            'most_common_count': most_common_count,
            'complex_colors_count': len(set(complex_colors)),
            'complex_colors_list': list(set(complex_colors)),
            'color_density': (total_color_mentions / total_tokens_no_punct) if total_tokens_no_punct > 0 else 0,
            'auto_detected_complex_colors': list(self.auto_detected_complex_colors),
        }
        return stats

    def print_results(self, stats, colors_found):
        print(f"\nОБЩАЯ СТАТИСТИКА ТЕКСТА:")
        print(f"Всего слов: {stats['total_words']}")
        print(f"Уникальных слов(лемм): {stats['unique_words']}")
        print(f"Общее упоминание цветов: {stats['total_color_mentions']}")
        print(f"Уникальных цветов: {stats['unique_colors']}")
        print(f"Плотность цветов: {stats['color_density']:.4f}")
        print(f"\nСТАТИСТИКА ПО ЦВЕТАМ:")
        top = stats.get('most_common_colors') or []
        if top:
            if len(top) == 1:
                print(f"Самый частый цвет: '{top[0]}' ({stats['most_common_count']} упоминаний)")
            else:
                joined = ", ".join(top)
                print(f"Самые частые цвета: {joined} ({stats['most_common_count']} упоминаний каждый)")
        print(f"\nСЛОЖНЫЕ ЦВЕТА ({stats['complex_colors_count']}):")
        if stats['complex_colors_list']:
            for color in sorted(stats['complex_colors_list']):
                count = colors_found.count(color)
                source = "(авто)" if color in stats['auto_detected_complex_colors'] else "(словарь)"
                print(f"  - {color} {source}: {count} упоминаний")
        else:
            print("Не найдено")
        print(f"\nЧАСТОТНОСТЬ ЦВЕТОВ:")
        print(f"{' ЦВЕТ':25} {' АБСОЛ':6} {' ОТН(слов)':10} {' ОТН(цвет)':10}")
        print(f"{'-' * 25} {'-' * 6} {'-' * 10} {'-' * 10}")
        if stats['color_frequency_absolute']:
            for color, abs_freq in sorted(stats['color_frequency_absolute'].items(), key=lambda x: x[1], reverse=True):
                rel_to_words = stats['color_frequency_relative_to_words'].get(color, 0)
                rel_to_colors = stats['color_frequency_relative_to_colors'].get(color, 0)
                source = " (авто)" if color in stats['auto_detected_complex_colors'] else ""
                print(f"{color:25} {abs_freq:6} {rel_to_words:10.4f} {rel_to_colors:10.4f}{source}")
            print(f"{'-' * 25} {'-' * 6} {'-' * 10} {'-' * 10}")
            print(f"{'ВСЕГО ЦВЕТОВ:':25} {stats['total_color_mentions']:6}")
        else:
            print("Цвета не найдены")

    def _safe_name(self, path_like: str) -> str:
        return re.sub(r'[^0-9A-Za-zА-Яа-яЁё._-]+', '_', path_like)

    def _save_path(self, filename: str) -> str:
        return str(self.output_dir / filename)

    def _base_from_textname(self, text_name: str) -> str:
        p = Path(text_name) if text_name else Path("text")
        return self._safe_name(p.stem)

    def _display_name(self, maybe_title_or_path: str) -> str:
        tail = str(maybe_title_or_path).split(":")[-1].strip()
        return Path(tail).stem

    # вывод в файл и csv
    def _save_colors_outputs(self, text_name: str, colors_found: list, stats: dict):
        base = self._base_from_textname(text_name)
        unique_ordered = []
        seen = set()
        for c in colors_found:
            if c not in seen:
                seen.add(c)
                unique_ordered.append(c)
        raw_path = self._save_path(f"colors_raw_{base}.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            for c in unique_ordered:
                f.write(f"{c}\n")
        print(f"Список уникальных цветов сохранён: {raw_path}")
        if stats and stats.get("color_frequency_absolute"):
            rows = []
            abs_map = stats["color_frequency_absolute"]
            rel_w = stats.get("color_frequency_relative_to_words", {})
            rel_c = stats.get("color_frequency_relative_to_colors", {})
            for color, cnt in sorted(abs_map.items(), key=lambda x: x[1], reverse=True):
                rows.append({
                    "Цвет": color,
                    "Абсолютная частота": cnt,
                    "Относит. к словам": round(rel_w.get(color, 0.0), 6),
                    "Относит. к цветам": round(rel_c.get(color, 0.0), 6),
                })
            df = pd.DataFrame(rows)
            freq_path = self._save_path(f"colors_freq_{base}.csv")
            df.to_csv(freq_path, index=False, encoding="utf-8-sig")
            print(f"Частоты цветов сохранены: {freq_path}")

    #  графики частоты по одному тексту
    def plot_color_frequency(self, stats, title="Частотность цветов", save_as: str = None, show: bool = False):
        if not stats['color_frequency_absolute']:
            print("Нет данных для построения графика")
            return
        colors, counts = zip(*sorted(stats['color_frequency_absolute'].items(), key=lambda x: x[1], reverse=True))
        color_map = self._get_color_map_for_labels(colors)
        bar_colors = [color_map[c] for c in colors]
        fname = self._display_name(title)
        title_clean = f"Частотность цветов: {fname}"
        plt.figure(figsize=(10, 5))
        plt.bar(colors, counts, color=bar_colors)
        plt.title(title_clean)
        plt.ylabel("Абсолютная частота упоминаний")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_as is None:
            save_as = self._safe_name(title_clean) + ".png"
        save_as = self._save_path(save_as)
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График сохранён: {save_as}")
        if show:
            plt.figure(figsize=(10, 5))
            plt.bar(colors, counts, color=bar_colors)
            plt.title(title_clean)
            plt.ylabel("Абсолютная частота упоминаний")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    #  сравнительная таблица
    def _print_comparison_table(self, results: dict):
        if not results:
            print("Нет данных для сравнительной таблицы")
            return
        rows = []
        for name, st in results.items():
            if not st:
                continue
            rows.append({
                'Текст': name,
                'Слова': st['total_words'],
                'Уник. слова': st['unique_words'],
                'Упомин. цветов': st['total_color_mentions'],
                'Уник. цветов': st['unique_colors'],
                'Плотность': round(st['color_density'], 4),
                'Топ-цвет(а)': ", ".join(st.get('most_common_colors') or [])
            })
        if not rows:
            print("Нет валидных строк для таблицы")
            return
        df = pd.DataFrame(rows)
        print(f"\n{'=' * 60}")
        print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))
        csv_path = self._save_path("comparison_table.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV со сравнительной таблицей сохранён: {csv_path}")

    # сравнительный график по нескольким текстам
    def plot_comparison(self, results: dict, title: str = "Сравнение плотности цветовой лексики",save_as: str = "comparison_density.png", show: bool = False):
        if not results:
            print("Нет результатов для сравнения")
            return
        names_raw = list(results.keys())
        names = [Path(n).stem for n in names_raw]
        densities = [r['color_density'] for r in results.values()]
        color_map = self._get_color_map_for_labels(names)
        bar_colors = [color_map[n] for n in names]
        plt.figure(figsize=(max(8, len(names) * 0.9), 5))
        plt.bar(names, densities, color=bar_colors)
        plt.title(title)
        plt.ylabel("Плотность (упоминания / слова)")
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        if not save_as:
            save_as = self._safe_name(title) + ".png"
        save_as = self._save_path(save_as)
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График сравнений сохранён: {save_as}")
        if show:
            plt.figure(figsize=(max(8, len(names) * 0.9), 5))
            plt.bar(names, densities, color=bar_colors )
            plt.title(title)
            plt.ylabel("Плотность (упоминания / слова)")
            plt.xticks(rotation=25, ha='right')
            plt.tight_layout()
            plt.show()

    #  анализ одного текста
    def analyze_text(self, text, text_name):
        print(f"\n{'=' * 60}")
        print(f"АНАЛИЗ ТЕКСТА: {text_name}")
        print(f"{'=' * 60}")
        if not self.color_dict['full_colors']:
            print("Словарь цветов не загружен! Анализ невозможен")
            return None
        processed_text = self.preprocess_text(text)
        tokens = self.morphological_analysis(processed_text)
        # print("\nТОКЕНЫ ДЛЯ ОТЛАДКИ:")
        # for i, token in enumerate(tokens):
        #     print(f"  {i:2}: '{token['text']:15} -> лемма: '{token['lemma']:15} основа: '{token['stem']:15} POS: {token['pos']}")
        colors_found = self.extract_colors(tokens)
        # print(f"\nНайденные цвета: {colors_found}")
        # print(f"Общее количество упоминаний: {len(colors_found)}")
        # if self.auto_detected_complex_colors:
        #     print(f"Автоматически обнаруженные сложные цвета: {list(self.auto_detected_complex_colors)}")
        stats = self.calculate_statistics(processed_text, colors_found, tokens)
        self.print_results(stats, colors_found)
        self._save_colors_outputs(text_name, colors_found, stats)
        return stats

    # анализ нескольких текстов
    def analyze_multiple_texts(self, file_paths):
        results = {}
        for path in file_paths:
            path = path.strip()
            if not path:
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Ошибка чтения файла '{path}': {e}")
                continue
            stats = self.analyze_text(text, text_name=path)
            results[path] = stats
            base = self._base_from_textname(path)
            self.plot_color_frequency(stats, title=f"Частотность цветов: {path}", save_as=f"freq_{base}.png",show=True)
        if not results:
            print("Нет валидных результатов для сравнения.")
            return
        self._print_comparison_table(results)
        self.plot_comparison(results, title="Сравнение плотности цветовой лексики", save_as="comparison_density.png",show=True)

    # с позициями для цветовой дорожки
    def extract_colors_with_positions(self, tokens):
        hits = []
        if not self.color_dict['full_colors']:
            return hits
        for idx, t in enumerate(tokens):
            txt = t['text']
            lem = t['lemma']
            st = t['stem']
            col = None
            if lem in self.color_dict['full_colors']:
                col = lem
            elif txt in self.color_dict['full_colors']:
                col = txt
            elif st in self.color_dict['color_stems']:
                base = self.find_color_by_stem(st)
                if base:
                    col = base
            elif '-' in txt:
                if txt in self.color_dict['complex_colors']:
                    col = txt
                else:
                    normalized = self.normalize_multi_hyphen_color(txt, lem)
                    if normalized:
                        col = normalized
                    else:
                        parts = txt.split('-')
                        if len(parts) == 2:
                            lemma_parts = lem.split('-') if lem else parts
                            last_lemma = lemma_parts[-1]
                            if self.is_simple_color_lemma(last_lemma):
                                col = f"{parts[0]}-{last_lemma}"
                            else:
                                tail = self.find_simple_color_for_part(last_lemma)
                                if tail:
                                    col = f"{parts[0]}-{tail}"
                                    self.auto_detected_complex_colors.add(col)
            if col:
                hits.append((idx, col))
        return hits

    #увеличиваем палитру, чтобы было больше цветов
    def _combined_cmap_colors(self, cmap_names, per_cmap=20):
        all_cols = []
        for name in cmap_names:
            cmap = plt.get_cmap(name, per_cmap)
            all_cols.extend([cmap(i) for i in range(cmap.N)])
        uniq, seen = [], set()
        for r, g, b, a in all_cols:
            key = (round(r, 3), round(g, 3), round(b, 3), round(a, 3))
            if key not in seen:
                seen.add(key)
                uniq.append((r, g, b, a))
        return uniq

    def _sample_from_continuous(self, name, n, start=0.02, end=0.98):
        cmap = plt.get_cmap(name)
        if n <= 1:
            return [cmap((start + end) / 2)]
        return [cmap(start + (end - start) * i / (n - 1)) for i in range(n)]

    def _make_huge_palette(self, target_n=200):
        base = self._combined_cmap_colors(["tab20", "tab20b", "tab20c", "Set3", "Paired"],per_cmap=20,)
        if len(base) < target_n:
            need = target_n - len(base)
            extra = self._sample_from_continuous("nipy_spectral", need, 0.02, 0.98)
            base.extend(extra)
            uniq, seen = [], set()
            for r, g, b, a in base:
                key = (round(r, 3), round(g, 3), round(b, 3), round(a, 3))
                if key not in seen:
                    seen.add(key)
                    uniq.append((r, g, b, a))
            base = uniq
        return base

    def _get_color_map_for_labels(self, labels):
        order = list(dict.fromkeys(labels))
        palette = self._make_huge_palette(max(200, len(order) + 20))
        return {name: palette[i % len(palette)] for i, name in enumerate(order)}

    # цветовая дорожка
    def plot_color_timeline(self,hits,text_name,save_as: str = None,show: bool = False,top_legend: int = 12, linewidth: float = 1.8,alpha: float = 0.9,save_tokens_csv: bool = True):
        if not hits:
            print("Нет попаданий цветов для построения цветовой дорожки")
            return
        positions = [int(pos) for pos, i in hits]
        labels = [col for i, col in hits]
        freq = Counter(labels)
        top_colors = [c for c, i in freq.most_common(top_legend)]
        color_map = self._get_color_map_for_labels(labels)
        if save_tokens_csv:
            df = pd.DataFrame([{"Позиция(токен)": int(pos), "Цвет": name} for pos, name in hits])
            print(df.to_string(index=False))
            base = self._base_from_textname(text_name)
            csv_path = self._save_path(f"timeline_tokens_{base}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"CSV-дорожка сохранёна: {csv_path}")
        min_pos, max_pos = min(positions), max(positions)
        span = max(1, max_pos - min_pos)
        pad = max(1, int(round(span * 0.05)))
        width = max(10, min(30, int(len(positions) * 0.0003) + 10))
        fig, ax = plt.subplots(figsize=(width, 3.2))
        for x, name in zip(positions, labels):
            ax.vlines(x, 0, 1, colors=[color_map[name]],
                      linewidth=linewidth, alpha=alpha, zorder=5)
        fname = self._display_name(text_name)
        ax.set_title(f"Цветовая дорожка: {fname}")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Позиция по токенам")
        ax.set_xlim(min_pos - pad, max_pos + pad)
        ax.margins(x=0.02)
        for side in ("right", "left", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_alpha(0.3)
        from matplotlib.lines import Line2D
        legend_handles = [Line2D([0], [0], color=color_map[c], lw=4, label=f"{c} ({freq[c]})")for c in top_colors]
        if legend_handles:
            ax.legend(handles=legend_handles,title="Топ цветов",frameon=False,ncol=1,loc="center left",bbox_to_anchor=(1.02, 0.5),fontsize=8,title_fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        base = self._base_from_textname(text_name)
        out = self._save_path(save_as or f"timeline_{base}.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        print(f"График цветовой дорожки сохранён: {out}")

    def analyze_timeline(self, text, text_name, show=True):
        print(f"\n{'=' * 60}")
        print(f"ЦВЕТОВАЯ ДОРОЖКА: {text_name}")
        print(f"{'=' * 60}")
        processed_text = self.preprocess_text(text)
        tokens = self.morphological_analysis(processed_text)
        hits = self.extract_colors_with_positions(tokens)
        if not hits:
            print("Цветов не обнаружено — строить дорожку нечего")
            return
        self.plot_color_timeline(hits, text_name=text_name, show=show)

print("Выберите тип словаря:")
print("1 — По умолчанию")
print("2 — Свой словарь")
while True:
    mode_dictionary = input("Введите номер типа: ").strip()
    if mode_dictionary in ("1", "2"):
        break
    print("Некорректный ввод. Введите 1 или 2\n")
print()
if mode_dictionary == "1":
    analyzer = ColorAnalyzer("dictionary.txt")
if mode_dictionary == "2":
    dictionary = input("Введите путь к файлу-словарю: ")
    analyzer = ColorAnalyzer(dictionary)
print("\nВыберите режим:")
print("1 — Тестовый (один файл)")
print("2 — Основной (несколько файлов)")
print("3 — Цветовая дорожка (один файл)")
mode = input("Введите номер режима: ").strip()
# тестовый режим
if mode == "1":
    while True:
        path = input("Введите путь к файлу: ").strip()
        try:
            with open(path, 'r', encoding='utf-8') as reader:
                text_f = reader.read()
            stats = analyzer.analyze_text(text_f, text_name=path)
            base = analyzer._base_from_textname(path)
            analyzer.plot_color_frequency(stats,title=f"Частотность цветов: {path}",save_as=f"freq_{base}.png",show=True)
            break
        except Exception as e:
            print(f"Ошибка чтения: {e}")
            print("Попробуйте снова")
# основной режим
elif mode == "2":
    print("Введите пути к файлам через запятую")
    inp = input("Файлы: ").strip()
    file_paths = [p.strip() for p in inp.split(",") if p.strip()]
    if not file_paths:
        print("Список файлов пуст. Завершение")
    else:
        analyzer.analyze_multiple_texts(file_paths)
# цветовая дорожка по одному файлу
elif mode == "3":
    path = input("Введите путь к файлу: ").strip()
    try:
        with open(path, 'r', encoding='utf-8') as reader:
            text_f = reader.read()
        analyzer.analyze_timeline(text_f, text_name=path, show=True)
    except Exception as e:
        print(f"Ошибка чтения: {e}")
else:
    print("Неизвестный режим. Завершение")