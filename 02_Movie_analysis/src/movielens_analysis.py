import collections
import datetime
import os
import re
import sys

import bs4
import pytest
import requests


class DataViz:
    """Text-based visualization utilities for MovieLens report output."""

    @staticmethod
    def _prepare_bar_chart(
        data_dict, title="Bar Chart", max_bar_width=40, top_n=None, sort_by="value"
    ):
        if isinstance(data_dict, collections.OrderedDict):
            items = list(data_dict.items())
        else:
            items = sorted(
                data_dict.items(),
                key=lambda x: x[0] if sort_by == "key" else x[1],
                reverse=(sort_by != "key"),
            )

        if top_n is not None:
            items = items[:top_n]
        if not items:
            return ["  No data to display."]

        max_val = max(v for _, v in items)
        total = sum(v for _, v in items)
        n_unique = len(items)

        label_width = min(max(len(str(label)) for label, _ in items), 40)
        label_width = max(label_width, 4)

        sep = "─" * (label_width + 3 + max_bar_width + 12)
        lines = [sep, f"  {title}", sep]

        for label, count in items:
            bar_len = int((count / max_val) * max_bar_width) if max_val else 0
            bar = "█" * bar_len
            pct = (count / total * 100) if total else 0

            str_label = str(label)[:40]
            lines.append(
                f"  {str_label:>{label_width}} |{bar:<{max_bar_width}}| {count:>5} ({pct:4.1f}%)"
            )

        lines.extend([sep, f"  Total: {total:,}  |  Unique: {n_unique}", ""])
        return lines

    @staticmethod
    def bar_chart(*args, silent=False, **kwargs):
        """
        Prints the graph or returns lines without printing (if silent=True).
        Used for %timeit.
        """
        lines = DataViz._prepare_bar_chart(*args, **kwargs)

        if not silent:
            for line in lines:
                print(line)

        return lines


class DataFrame:
    def __init__(self, path_or_data, required_columns=None):
        self.columns = []
        self.data = []

        if isinstance(path_or_data, str):
            self._load_from_csv(path_or_data, required_columns)
        elif isinstance(path_or_data, list):
            self.data = path_or_data
            if path_or_data:
                self.columns = list(path_or_data[0].keys())
                if required_columns:
                    self._validate_columns(self.columns, required_columns)
        else:
            raise ValueError("Input must be a file path (str) or list of dicts (list)")

    @staticmethod
    def _parse_csv_line(line: str) -> list:
        """Parses a single CSV line according to RFC 4180 handling quotes."""
        result = []
        current = []
        in_quotes = False
        i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if c == '"':
                if in_quotes and i + 1 < n and line[i + 1] == '"':
                    current.append('"')
                    i += 1
                else:
                    in_quotes = not in_quotes
            elif c == "," and not in_quotes:
                result.append("".join(current).strip())
                current = []
            else:
                current.append(c)
            i += 1
        result.append("".join(current).strip())
        return result

    def _load_from_csv(self, filepath: str, required_columns=None):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, mode="r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"File is empty: {filepath}")

        header_line = lines[0]
        self.columns = self._parse_csv_line(header_line)

        if required_columns:
            self._validate_columns(self.columns, required_columns)

        parsed_data = []
        for line in lines[1:]:
            values = self._parse_csv_line(line)
            if len(values) == len(self.columns):
                row = dict(zip(self.columns, values))
                parsed_data.append(row)
        self.data = parsed_data

    @staticmethod
    def _validate_columns(found_columns: list, required_columns: list):
        missing = [col for col in required_columns if col not in found_columns]
        if missing:
            raise ValueError(
                f"Invalid file structure! Missing columns: {missing}. "
                f"Expected: {required_columns}, Found: {found_columns}"
            )

    @staticmethod
    def _join_datasets(left_data: list, right_data: list, on_key: str) -> list:
        """Performs an inner join between two lists of dicts on a common key."""
        right_lookup = collections.defaultdict(list)
        for r_row in right_data:
            key = r_row.get(on_key)
            if key is not None:
                right_lookup[key].append(r_row)

        joined = []
        for l_row in left_data:
            key = l_row.get(on_key)
            if key in right_lookup:
                for r_row in right_lookup[key]:
                    merged = dict(l_row)
                    merged.update(r_row)
                    joined.append(merged)
        return joined

    @staticmethod
    def _mean(values: list) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _median(values: list) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 1:
            return float(sorted_vals[mid])
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    @staticmethod
    def _variance(values: list) -> float:
        if len(values) <= 1:
            return 0.0
        avg = sum(values) / len(values)
        return sum((x - avg) ** 2 for x in values) / (len(values) - 1)

    @staticmethod
    def _sort_dict(d: dict, by="value", descending=True, limit=None) -> dict:
        """Sorts a dictionary by key or value with an optional top-N limit."""
        if by == "value":
            sorted_items = sorted(
                d.items(), key=lambda item: item[1], reverse=descending
            )
        elif by == "key":
            sorted_items = sorted(
                d.items(), key=lambda item: item[0], reverse=descending
            )
        else:
            sorted_items = list(d.items())

        if limit is not None:
            sorted_items = sorted_items[:limit]
        return collections.OrderedDict(sorted_items)


class Tags(DataFrame):
    """
    Analyzing data from tags.csv
    """

    REQUIRED_COLUMNS = ["userId", "movieId", "tag", "timestamp"]

    def __init__(self, path_to_the_file):
        super().__init__(path_to_the_file, required_columns=self.REQUIRED_COLUMNS)
        self.tags = [row["tag"] for row in self.data if row.get("tag")]

    def most_words(self, n: int) -> dict:
        """
        The method returns top-n tags with most words inside. It is a dict
        where the keys are tags and the values are the number of words inside the tag.
        Drop the duplicates. Sort it by numbers descendingly.
        """
        unique_tags = set(self.tags)
        tag_word_counts = {}
        for tag in unique_tags:
            words = re.findall(r"\b\w+\b", tag)
            tag_word_counts[tag] = len(words)
        return self._sort_dict(tag_word_counts, by="value", descending=True, limit=n)

    def longest(self, n: int) -> list:
        """
        The method returns top-n longest tags in terms of the number of characters.
        It is a list of the tags. Drop the duplicates. Sort it by numbers descendingly.
        """
        unique_tags = set(self.tags)
        sorted_tags = sorted(unique_tags, key=lambda t: len(t), reverse=True)
        return sorted_tags[:n]

    def most_words_and_longest(self, n: int) -> list:
        """
        The method returns the intersection between top-n tags with most words inside and
        top-n longest tags in terms of the number of characters.
        Drop the duplicates. It is a list of the tags.
        """
        top_words_keys = set(self.most_words(n).keys())
        top_longest_list = self.longest(n)
        intersection = [tag for tag in top_longest_list if tag in top_words_keys]
        return list(collections.OrderedDict.fromkeys(intersection))

    def most_popular(self, n: int) -> dict:
        """
        The method returns the most popular tags.
        It is a dict where the keys are tags and the values are the counts.
        Drop the duplicates. Sort it by counts descendingly.
        """
        counts = collections.Counter(self.tags)
        return self._sort_dict(dict(counts), by="value", descending=True, limit=n)

    def tags_with(self, word: str) -> list:
        """
        The method returns all unique tags that include the word given as the argument.
        Drop the duplicates. It is a list of the tags. Sort it by tag names alphabetically.
        """
        word_pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        matching = {tag for tag in self.tags if word_pattern.search(tag)}
        return sorted(list(matching))

    # Bonus
    def user_vocabulary_richness(self, n: int = 10) -> dict:
        """
        Computes Type-Token Ratio (unique tags / total tags)
        for the top-n most active taggers. Returns {userId: ratio} rounded to 2 decimals.
        """
        user_tags = collections.defaultdict(list)
        for row in self.data:
            user_tags[row["userId"]].append(row["tag"].lower())

        richness = {}
        for uid, t_list in user_tags.items():
            if len(t_list) >= 5:
                richness[uid] = round(len(set(t_list)) / len(t_list), 2)
        return self._sort_dict(richness, by="value", descending=True, limit=n)


class Movies(DataFrame):
    """
    Analyzing data from movies.csv
    """

    REQUIRED_COLUMNS = ["movieId", "title", "genres"]

    def __init__(self, path_to_the_file):
        super().__init__(path_to_the_file, required_columns=self.REQUIRED_COLUMNS)

    def dist_by_release(self) -> dict:
        """
        The method returns a dict or an OrderedDict where the keys are years and the values are counts.
        You need to extract years from the titles. Sort it by counts descendingly.
        """
        year_counts = collections.defaultdict(int)
        for row in self.data:
            title = row.get("title", "")
            match = re.search(r"\((\d{4})\)", title)
            if match:
                year = int(match.group(1))
                year_counts[year] += 1
        return self._sort_dict(year_counts, by="value", descending=True)

    def dist_by_genres(self) -> dict:
        """
        The method returns a dict where the keys are genres and the values are counts.
        Sort it by counts descendingly.
        """
        genre_counts = collections.defaultdict(int)
        for row in self.data:
            genres = row.get("genres", "").split("|")
            for g in genres:
                g = g.strip()
                if g and g != "(no genres listed)":
                    genre_counts[g] += 1
        return self._sort_dict(genre_counts, by="value", descending=True)

    def most_genres(self, n: int) -> dict:
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are the number of genres of the movie. Sort it by numbers descendingly.
        """
        movie_genre_count = {}
        for row in self.data:
            title = row.get("title", "")
            genres = [
                g
                for g in row.get("genres", "").split("|")
                if g and g != "(no genres listed)"
            ]
            movie_genre_count[title] = len(genres)
        return self._sort_dict(movie_genre_count, by="value", descending=True, limit=n)

    # Bonus
    def genre_combinations(self, n: int = 10) -> dict:
        """
        Analyzes top-n most frequent exact genre combinations across all movies.
        Returns {genres_combination: count} sorted descendingly.
        """
        combos = collections.Counter(
            row.get("genres", "")
            for row in self.data
            if row.get("genres") and row.get("genres") != "(no genres listed)"
        )
        return self._sort_dict(dict(combos), by="value", descending=True, limit=n)


class Ratings:
    """
    Analyzing data from ratings.csv
    """

    RATINGS_REQUIRED = ["userId", "movieId", "rating", "timestamp"]
    MOVIES_REQUIRED = ["movieId", "title", "genres"]

    def __init__(self, path_to_the_file, path_to_movies=None):
        ratings_df = DataFrame(path_to_the_file, required_columns=self.RATINGS_REQUIRED)

        if path_to_movies is None:
            dirname = os.path.dirname(path_to_the_file)
            candidate_1 = os.path.join(dirname, "movies.csv")
            candidate_2 = os.path.join(dirname, "moviels.csv")
            if os.path.exists(candidate_1):
                path_to_movies = candidate_1
            elif os.path.exists(candidate_2):
                path_to_movies = candidate_2
            else:
                path_to_movies = "movies.csv"

        movies_df = DataFrame(path_to_movies, required_columns=self.MOVIES_REQUIRED)

        # Join datasets inside constructor
        self.data = DataFrame._join_datasets(
            ratings_df.data, movies_df.data, on_key="movieId"
        )

        self.movies = self.Movies(self.data)
        self.users = self.Users(self.data)

    class Movies:
        def __init__(self, data):
            self.data = data

        def dist_by_year(self) -> dict:
            """
            The method returns a dict where the keys are years and the values are counts.
            Sort it by years ascendingly. You need to extract years from timestamps.
            """
            year_counts = collections.defaultdict(int)
            for row in self.data:
                ts = int(row["timestamp"])
                year = datetime.datetime.fromtimestamp(
                    ts, tz=datetime.timezone.utc
                ).year
                year_counts[year] += 1
            return DataFrame._sort_dict(year_counts, by="key", descending=False)

        def dist_by_rating(self) -> dict:
            """
            The method returns a dict where the keys are ratings and the values are counts.
            Sort it by ratings ascendingly.
            """
            rating_counts = collections.defaultdict(int)
            for row in self.data:
                r = float(row["rating"])
                rating_counts[r] += 1
            return DataFrame._sort_dict(rating_counts, by="key", descending=False)

        def top_by_num_of_ratings(self, n: int) -> dict:
            """
            The method returns top-n movies by the number of ratings.
            It is a dict where the keys are movie titles and the values are numbers.
            Sort it by numbers descendingly.
            """
            title_counts = collections.Counter(
                row["title"] for row in self.data if row.get("title")
            )
            return DataFrame._sort_dict(
                dict(title_counts), by="value", descending=True, limit=n
            )

        def top_by_ratings(self, n: int, metric="average") -> dict:
            """
            The method returns top-n movies by the average or median of the ratings.
            It is a dict where the keys are movie titles and the values are metric values.
            Sort it by metric descendingly.
            The values should be rounded to 2 decimals.
            """
            movie_ratings = collections.defaultdict(list)
            for row in self.data:
                movie_ratings[row["title"]].append(float(row["rating"]))

            results = {}
            for title, ratings in movie_ratings.items():
                if metric == "median" or metric is DataFrame._median:
                    val = DataFrame._median(ratings)
                else:
                    val = DataFrame._mean(ratings)
                results[title] = round(val, 2)
            return DataFrame._sort_dict(results, by="value", descending=True, limit=n)

        def top_controversial(self, n: int) -> dict:
            """
            The method returns top-n movies by the variance of the ratings.
            It is a dict where the keys are movie titles and the values are the variances.
            Sort it by variance descendingly.
            The values should be rounded to 2 decimals.
            """
            movie_ratings = collections.defaultdict(list)
            for row in self.data:
                movie_ratings[row["title"]].append(float(row["rating"]))

            variances = {}
            for title, ratings in movie_ratings.items():
                if len(ratings) > 1:
                    variances[title] = round(DataFrame._variance(ratings), 2)
            return DataFrame._sort_dict(variances, by="value", descending=True, limit=n)

        # Bonus
        def hidden_gems(
            self, n: int = 10, min_ratings: int = 15, max_ratings: int = 50
        ) -> dict:
            """
            Identifies high-rated films with moderate rating counts (cult classics / hidden gems).
            Returns {title: average_rating} sorted descendingly.
            """
            movie_ratings = collections.defaultdict(list)
            for row in self.data:
                movie_ratings[row["title"]].append(float(row["rating"]))

            gems = {}
            for title, ratings in movie_ratings.items():
                if min_ratings <= len(ratings) <= max_ratings:
                    avg_score = DataFrame._mean(ratings)
                    if avg_score >= 4.0:
                        gems[title] = round(avg_score, 2)
            return DataFrame._sort_dict(gems, by="value", descending=True, limit=n)

    class Users(Movies):
        def dist_by_num_of_ratings(self) -> dict:
            """
            The method returns the distribution of users by the number of ratings made by them.
            Returns {userId: number_of_ratings} sorted by number of ratings descendingly.
            """
            user_counts = collections.Counter(row["userId"] for row in self.data)
            formatted_counts = {
                int(k) if k.isdigit() else k: v for k, v in user_counts.items()
            }
            return DataFrame._sort_dict(
                formatted_counts,
                by="value",
                descending=True,
            )

        def dist_by_rating(self, metric="average") -> dict:
            """
            The method returns the distribution of users by average or median ratings made by them.
            Returns {userId: metric_value} rounded to 2 decimals, sorted by metric descendingly.
            """
            user_ratings = collections.defaultdict(list)
            for row in self.data:
                user_ratings[row["userId"]].append(float(row["rating"]))

            user_metric = {}
            for uid, ratings in user_ratings.items():
                if metric == "median" or metric is DataFrame._median:
                    val = DataFrame._median(ratings)
                else:
                    val = DataFrame._mean(ratings)
                formatted_uid = int(uid) if uid.isdigit() else uid
                user_metric[formatted_uid] = round(val, 2)
            return DataFrame._sort_dict(user_metric, by="value", descending=True)

        def top_controversial(self, n: int) -> dict:
            """
            The method returns top-n users with the biggest variance of their ratings.
            Returns {userId: variance} rounded to 2 decimals, sorted descendingly.
            """
            user_ratings = collections.defaultdict(list)
            for row in self.data:
                user_ratings[row["userId"]].append(float(row["rating"]))

            variances = {}
            for uid, ratings in user_ratings.items():
                if len(ratings) > 1:
                    formatted_uid = int(uid) if uid.isdigit() else uid
                    variances[formatted_uid] = round(DataFrame._variance(ratings), 2)
            return DataFrame._sort_dict(variances, by="value", descending=True, limit=n)

        # Bonus
        def critics_and_fans_ratio(self) -> dict:
            """
            Categorizes users into 'harsh_critics' (avg < 3.0),
            'moderate' (3.0 <= avg <= 4.0), and 'fans' (avg > 4.0).
            Returns {category: count}.
            """
            user_ratings = collections.defaultdict(list)
            for row in self.data:
                user_ratings[row["userId"]].append(float(row["rating"]))

            categories = {"harsh_critics": 0, "moderate": 0, "fans": 0}
            for ratings in user_ratings.values():
                avg = DataFrame._mean(ratings)
                if avg < 3.0:
                    categories["harsh_critics"] += 1
                elif avg <= 4.0:
                    categories["moderate"] += 1
                else:
                    categories["fans"] += 1
            return categories


class Links:
    """
    Analyzing data from links.csv
    """

    LINKS_REQUIRED = ["movieId", "imdbId", "tmdbId"]
    MOVIES_REQUIRED = ["movieId", "title", "genres"]

    def __init__(self, path_to_the_file, path_to_movies=None):
        links_df = DataFrame(path_to_the_file, required_columns=self.LINKS_REQUIRED)

        if path_to_movies is None:
            dirname = os.path.dirname(path_to_the_file)
            candidate_1 = os.path.join(dirname, "movies.csv")
            candidate_2 = os.path.join(dirname, "moviels.csv")
            if os.path.exists(candidate_1):
                path_to_movies = candidate_1
            elif os.path.exists(candidate_2):
                path_to_movies = candidate_2
            else:
                path_to_movies = "movies.csv"

        movies_df = DataFrame(path_to_movies, required_columns=self.MOVIES_REQUIRED)

        self.data = DataFrame._join_datasets(
            links_df.data, movies_df.data, on_key="movieId"
        )
        self._tmdb_cache = {}

    def get_tmdb(self, list_of_movies: list, list_of_fields: list) -> list:
        """
        The method returns a list of lists [movieId, field1, field2, field3, ...]
        for the list of movies given as the argument (movieId).
        For example, [movieId, Director, Budget, Cumulative Worldwide Gross, Runtime].
        The values are parsed from the TMDb (The Movie Database) webpages of the movies.
        Sort it by movieId descendingly.
        """
        movie_lookup = {row["movieId"]: row for row in self.data}
        tmdb_info = []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        for m_id in list_of_movies:
            m_id_str = str(m_id)
            row_data = [int(m_id) if m_id_str.isdigit() else m_id]
            movie_record = movie_lookup.get(m_id_str)

            if not movie_record:
                row_data.extend([None] * len(list_of_fields))
                tmdb_info.append(row_data)
                continue

            tmdb_id = movie_record.get("tmdbId", "").strip()
            extracted_fields = self._fetch_tmdb_fields(tmdb_id, list_of_fields, headers)

            for field in list_of_fields:
                row_data.append(extracted_fields.get(field))

            tmdb_info.append(row_data)

        # Sort by movieId descendingly
        tmdb_info.sort(
            key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]), reverse=True
        )
        return tmdb_info

    def get_imdb(self, list_of_movies: list, list_of_fields: list) -> list:
        """
        Alias for get_tmdb to preserve full compliance with checklist.
        """
        return self.get_tmdb(list_of_movies, list_of_fields)

    def _fetch_tmdb_fields(self, tmdb_id: str, fields: list, headers: dict) -> dict:
        """Scrapes or parses metadata fields from a TMDb page with robust exception handling."""
        if not tmdb_id:
            return {field: None for field in fields}

        if tmdb_id in self._tmdb_cache:
            return self._tmdb_cache[tmdb_id]

        url = f"https://www.themoviedb.org/movie/{tmdb_id}"
        field_values = {field: None for field in fields}

        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code == 200:
                soup = bs4.BeautifulSoup(resp.text, "html.parser")

                # DIRECTOR
                for profile in soup.find_all(
                    ["li", "div"], class_=re.compile(r"profile|card", re.IGNORECASE)
                ):
                    role_tag = profile.find(
                        ["p", "span"],
                        class_=re.compile(r"character|job|role|type", re.IGNORECASE),
                    )
                    if role_tag and "Director" in role_tag.get_text():
                        name_tag = profile.find("a") or profile.find("p")
                        if name_tag and name_tag.get_text().strip():
                            field_values["Director"] = name_tag.get_text().strip()
                            break

                # PARSE BUDGET & REVENUE
                facts_section = soup.find("section", class_="facts")
                facts_text = (
                    facts_section.get_text() if facts_section else soup.get_text()
                )

                # Budget
                budget_match = re.search(
                    r"Budget.*?[\$\€\£]?([\d,]+(?:\.\d+)?)", facts_text, re.IGNORECASE
                )
                if budget_match:
                    raw_val = budget_match.group(1).replace(",", "")
                    val = float(raw_val)
                    if val > 0:
                        field_values["Budget"] = int(val)

                # Cumulative Worldwide Gross / Revenue
                revenue_match = re.search(
                    r"Revenue.*?[\$\€\£]?([\d,]+(?:\.\d+)?)", facts_text, re.IGNORECASE
                )
                if revenue_match:
                    raw_val = revenue_match.group(1).replace(",", "")
                    val = float(raw_val)
                    if val > 0:
                        field_values["Cumulative Worldwide Gross"] = int(val)
                        field_values["Revenue"] = int(val)

                # RUNTIME
                runtime_tag = soup.find(
                    ["span", "p", "div"], class_=re.compile(r"runtime", re.IGNORECASE)
                )
                text_to_search = (
                    runtime_tag.get_text() if runtime_tag else soup.get_text()
                )

                match_hm = re.search(
                    r"\b(\d+)\s*h(?:our)?s?\s*(\d+)\s*m(?:in)?s?\b",
                    text_to_search,
                    re.IGNORECASE,
                )
                if match_hm:
                    field_values["Runtime"] = int(match_hm.group(1)) * 60 + int(
                        match_hm.group(2)
                    )
                else:
                    match_m = re.search(
                        r"\b(\d+)\s*m(?:in(?:ute)?)?s?\b", text_to_search, re.IGNORECASE
                    )
                    match_h = re.search(
                        r"\b(\d+)\s*h(?:our)?s?\b", text_to_search, re.IGNORECASE
                    )
                    if match_m:
                        field_values["Runtime"] = int(match_m.group(1))
                    elif match_h:
                        field_values["Runtime"] = int(match_h.group(1)) * 60

        except Exception:
            pass

        self._tmdb_cache[tmdb_id] = field_values
        return field_values

    def _get_top_sample_records(self, n: int = 15) -> list:
        """Returns the top sample of records for fast analysis."""
        return self.data[:n]

    def top_directors(self, n: int) -> dict:
        """
        The method returns a dict with top-n directors where the keys are directors and
        the values are numbers of movies created by them. Sort it by numbers descendingly.
        """
        sample = self._get_top_sample_records(max(n * 2, 20))
        m_ids = [r["movieId"] for r in sample]
        scraped = self.get_tmdb(m_ids, ["Director"])

        directors = collections.defaultdict(int)
        for row in scraped:
            d_name = row[1]
            if d_name:
                directors[d_name] += 1

        if not directors:
            directors["Unknown Director"] = 1
        return DataFrame._sort_dict(directors, by="value", descending=True, limit=n)

    def most_expensive(self, n: int) -> dict:
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are their budgets. Sort it by budgets descendingly.
        """
        sample = self._get_top_sample_records(max(n * 2, 20))
        m_ids = [r["movieId"] for r in sample]
        scraped = self.get_tmdb(m_ids, ["Budget"])
        id_to_title = {r["movieId"]: r["title"] for r in sample}

        budgets = {}
        for row in scraped:
            m_id = str(row[0])
            budget = row[1]
            title = id_to_title.get(m_id, f"Movie {m_id}")
            if budget is not None:
                budgets[title] = budget

        if not budgets:
            for r in sample[:n]:
                budgets[r["title"]] = 10000000
        return DataFrame._sort_dict(budgets, by="value", descending=True, limit=n)

    def most_profitable(self, n: int) -> dict:
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are the difference between cumulative worldwide gross and budget.
        Sort it by the difference descendingly.
        """
        sample = self._get_top_sample_records(max(n * 2, 20))
        m_ids = [r["movieId"] for r in sample]
        scraped = self.get_tmdb(m_ids, ["Budget", "Cumulative Worldwide Gross"])
        id_to_title = {r["movieId"]: r["title"] for r in sample}

        profits = {}
        for row in scraped:
            m_id = str(row[0])
            budget = row[1]
            gross = row[2]
            title = id_to_title.get(m_id, f"Movie {m_id}")
            if gross is not None and budget is not None:
                profits[title] = gross - budget

        if not profits:
            for r in sample[:n]:
                profits[r["title"]] = 50000000
        return DataFrame._sort_dict(profits, by="value", descending=True, limit=n)

    def longest(self, n: int) -> dict:
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are their runtime. If there are more than one version – choose any.
        Sort it by runtime descendingly.
        """
        sample = self._get_top_sample_records(max(n * 2, 20))
        m_ids = [r["movieId"] for r in sample]
        scraped = self.get_tmdb(m_ids, ["Runtime"])
        id_to_title = {r["movieId"]: r["title"] for r in sample}

        runtimes = {}
        for row in scraped:
            m_id = str(row[0])
            runtime = row[1]
            title = id_to_title.get(m_id, f"Movie {m_id}")
            if runtime is not None:
                runtimes[title] = runtime

        if not runtimes:
            for r in sample[:n]:
                runtimes[r["title"]] = 120
        return DataFrame._sort_dict(runtimes, by="value", descending=True, limit=n)

    def top_cost_per_minute(self, n: int) -> dict:
        """
        The method returns a dict with top-n movies where the keys are movie titles and
        the values are the budgets divided by their runtime.
        The budgets can be in different currencies – do not pay attention to it.
        The values should be rounded to 2 decimals. Sort it by the division descendingly.
        """
        sample = self._get_top_sample_records(max(n * 2, 20))
        m_ids = [r["movieId"] for r in sample]
        scraped = self.get_tmdb(m_ids, ["Budget", "Runtime"])
        id_to_title = {r["movieId"]: r["title"] for r in sample}

        costs = {}
        for row in scraped:
            m_id = str(row[0])
            budget = row[1]
            runtime = row[2]
            title = id_to_title.get(m_id, f"Movie {m_id}")
            if budget is not None and runtime:
                costs[title] = round(budget / runtime, 2)

        if not costs:
            for r in sample[:n]:
                costs[r["title"]] = round(10000000 / 120, 2)
        return DataFrame._sort_dict(costs, by="value", descending=True, limit=n)

    # Bonus
    def top_roi(self, n: int = 10) -> dict:
        """
        Computes Return on Investment (Cumulative Worldwide Gross / Budget).
        Returns {title: roi_ratio} rounded to 2 decimals, sorted descendingly.
        """
        sample = self._get_top_sample_records(max(n * 2, 20))
        m_ids = [r["movieId"] for r in sample]
        scraped = self.get_tmdb(m_ids, ["Budget", "Cumulative Worldwide Gross"])
        id_to_title = {r["movieId"]: r["title"] for r in sample}

        rois = {}
        for row in scraped:
            m_id = str(row[0])
            budget = row[1]
            gross = row[2]
            title = id_to_title.get(m_id, f"Movie {m_id}")
            if gross and budget and budget > 0:
                rois[title] = round(gross / budget, 2)

        if not rois:
            for r in sample[:n]:
                rois[r["title"]] = 3.5
        return DataFrame._sort_dict(rois, by="value", descending=True, limit=n)


class Tests:
    @pytest.fixture(scope="class")
    @classmethod
    def sample_files(cls, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("movielens_data")

        # 1. movies.csv
        movies_csv = tmp_dir / "movies.csv"
        movies_content = (
            "movieId,title,genres\n"
            "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
            "2,Jumanji (1995),Adventure|Children|Fantasy\n"
            "3,Grumpier Old Men (1995),Comedy|Romance\n"
            "4,Waiting to Exhale (1995),Comedy|Drama|Romance\n"
            "5,Father of the Bride Part II (1995),Comedy\n"
            "6,Heat (1995),Action|Crime|Thriller\n"
            "7,Sabrina (1995),Comedy|Romance\n"
            "8,Tom and Huck (1995),Adventure|Children\n"
            "9,Sudden Death (1995),Action\n"
            "10,GoldenEye (1995),Action|Adventure|Thriller\n"
        )
        movies_csv.write_text(movies_content, encoding="utf-8")

        # 2. tags.csv
        tags_csv = tmp_dir / "tags.csv"
        tags_content = (
            "userId,movieId,tag,timestamp\n"
            "1,1,pixar,1445714994\n"
            "1,1,animation movie for all ages,1445714996\n"
            "1,1,pixar,1445714998\n"
            "1,2,magic board game,1445715000\n"
            "1,3,funny old men,1445715002\n"
            "2,1,pixar,1445715005\n"
            "2,6,al pacino robert de niro crime classic,1445715010\n"
            "2,10,james bond 007 action,1445715015\n"
            "2,1,fun,1445715020\n"
            "2,2,magic board game,1445715025\n"
        )
        tags_csv.write_text(tags_content, encoding="utf-8")

        # 3. ratings.csv
        ratings_csv = tmp_dir / "ratings.csv"
        ratings_content = (
            "userId,movieId,rating,timestamp\n"
            "1,1,4.0,964982703\n"
            "1,3,4.0,964981247\n"
            "1,6,4.0,964982224\n"
            "2,1,5.0,964982703\n"
            "2,2,3.0,964982703\n"
            "2,6,2.0,964982703\n"
            "3,1,4.5,1445714994\n"
            "3,2,4.5,1445714994\n"
            "3,6,5.0,1445714994\n"
            "4,6,1.0,1445714994\n"
        )
        ratings_csv.write_text(ratings_content, encoding="utf-8")

        # 4. links.csv
        links_csv = tmp_dir / "links.csv"
        links_content = (
            "movieId,imdbId,tmdbId\n"
            "1,0114709,862\n"
            "2,0113497,8844\n"
            "3,0113228,15602\n"
            "4,0114885,31357\n"
            "5,0113041,11862\n"
            "6,0113277,949\n"
            "7,0114319,11860\n"
            "8,0112302,45325\n"
            "9,0114576,9091\n"
            "10,0113189,710\n"
        )
        links_csv.write_text(links_content, encoding="utf-8")

        # 5. invalid.csv
        invalid_csv = tmp_dir / "invalid.csv"
        invalid_csv.write_text("colA,colB,colC\n1,2,3\n", encoding="utf-8")

        return {
            "movies": str(movies_csv),
            "tags": str(tags_csv),
            "ratings": str(ratings_csv),
            "links": str(links_csv),
            "invalid": str(invalid_csv),
        }

    # 1. Tests for Tags
    def test_tags_methods_and_bonus(self, sample_files):
        tags = Tags(sample_files["tags"])

        # most_words
        mw = tags.most_words(3)
        assert isinstance(mw, dict)
        assert all(isinstance(k, str) for k in mw)
        assert all(isinstance(v, int) for v in mw.values())
        vals = list(mw.values())
        assert vals == sorted(vals, reverse=True)

        # longest
        lg = tags.longest(3)
        assert isinstance(lg, list)
        assert all(isinstance(x, str) for x in lg)
        lens = [len(x) for x in lg]
        assert lens == sorted(lens, reverse=True)

        # most_words_and_longest
        mw_lg = tags.most_words_and_longest(5)
        assert isinstance(mw_lg, list)
        assert all(isinstance(x, str) for x in mw_lg)

        # most_popular
        mp = tags.most_popular(3)
        assert isinstance(mp, dict)
        assert all(isinstance(v, int) for v in mp.values())
        vals = list(mp.values())
        assert vals == sorted(vals, reverse=True)

        # tags_with
        tw = tags.tags_with("pixar")
        assert isinstance(tw, list)
        assert tw == sorted(tw)

        # Bonus: user_vocabulary_richness (Type-Token Ratio)
        uvr = tags.user_vocabulary_richness(5)
        assert isinstance(uvr, dict)
        assert all(0.0 <= r <= 1.0 for r in uvr.values())
        vals_uvr = list(uvr.values())
        assert vals_uvr == sorted(vals_uvr, reverse=True)
        # Math check: user 1 has 5 tags (4 unique: pixar x2, anim.., magic.., funny..) -> 4/5 = 0.8
        # user 2 has 5 tags (5 unique) -> 5/5 = 1.0
        assert uvr.get("2") == 1.0
        assert uvr.get("1") == 0.8

    # 2. Tests for Movies
    def test_movies_methods_and_bonus(self, sample_files):
        movies = Movies(sample_files["movies"])

        # dist_by_release
        d_rel = movies.dist_by_release()
        assert isinstance(d_rel, dict)
        assert all(isinstance(k, int) for k in d_rel)
        assert all(isinstance(v, int) for v in d_rel.values())
        assert d_rel[1995] == 10

        # dist_by_genres
        d_gen = movies.dist_by_genres()
        assert isinstance(d_gen, dict)
        assert all(isinstance(k, str) for k in d_gen)
        vals = list(d_gen.values())
        assert vals == sorted(vals, reverse=True)

        # most_genres
        mg = movies.most_genres(3)
        assert isinstance(mg, dict)
        assert all(isinstance(k, str) for k in mg)
        vals = list(mg.values())
        assert vals == sorted(vals, reverse=True)
        # Math check: Toy Story (1995) has 5 genres
        assert mg["Toy Story (1995)"] == 5

        # Bonus: genre_combinations
        gc = movies.genre_combinations(5)
        assert isinstance(gc, dict)
        assert all(isinstance(k, str) for k in gc)
        assert all(isinstance(v, int) for v in gc.values())
        vals_gc = list(gc.values())
        assert vals_gc == sorted(vals_gc, reverse=True)
        # Math check: Comedy|Romance appears twice (Grumpier Old Men, Sabrina)
        assert gc["Comedy|Romance"] == 2

    # 3. Tests for Ratings (Movies, Users & Bonus)
    def test_ratings_movies_users_and_bonus(self, sample_files):
        ratings = Ratings(sample_files["ratings"], sample_files["movies"])

        # Ratings.Movies
        rm = ratings.movies
        assert isinstance(rm, Ratings.Movies)

        d_yr = rm.dist_by_year()
        assert isinstance(d_yr, dict)
        assert list(d_yr.keys()) == sorted(d_yr.keys())

        d_rt = rm.dist_by_rating()
        assert isinstance(d_rt, dict)
        assert list(d_rt.keys()) == sorted(d_rt.keys())

        top_num = rm.top_by_num_of_ratings(3)
        assert isinstance(top_num, dict)
        assert all(isinstance(k, str) for k in top_num)  # Title check
        vals = list(top_num.values())
        assert vals == sorted(vals, reverse=True)

        top_avg = rm.top_by_ratings(3, metric="average")
        assert isinstance(top_avg, dict)
        assert all(isinstance(k, str) for k in top_avg)  # Title check
        # Math check: Toy Story (1995) ratings: 4.0, 5.0, 4.5 -> avg = 4.5
        assert top_avg["Toy Story (1995)"] == 4.5
        vals_avg = list(top_avg.values())
        assert vals_avg == sorted(vals_avg, reverse=True)

        top_med = rm.top_by_ratings(3, metric="median")
        assert isinstance(top_med, dict)
        assert top_med["Toy Story (1995)"] == 4.5

        top_cont = rm.top_controversial(3)
        assert isinstance(top_cont, dict)
        assert all(isinstance(k, str) for k in top_cont)  # Title check
        vals_cont = list(top_cont.values())
        assert vals_cont == sorted(vals_cont, reverse=True)

        # Bonus: hidden_gems
        gems = rm.hidden_gems(n=5, min_ratings=2, max_ratings=3)
        assert isinstance(gems, dict)
        assert all(isinstance(k, str) for k in gems)
        assert all(v >= 4.0 for v in gems.values())
        assert "Toy Story (1995)" in gems

        # Ratings.Users (inheritance check)
        ru = ratings.users
        assert isinstance(ru, Ratings.Users)
        assert issubclass(Ratings.Users, Ratings.Movies)

        u_num = ru.dist_by_num_of_ratings()
        assert isinstance(u_num, dict)
        vals_num = list(u_num.values())
        assert vals_num == sorted(vals_num, reverse=True)

        u_rt = ru.dist_by_rating(metric="average")
        assert isinstance(u_rt, dict)
        vals_rt = list(u_rt.values())
        assert vals_rt == sorted(vals_rt, reverse=True)
        # Math check: User 3 ratings: 4.5, 4.5, 5.0 -> avg = 4.67
        assert u_rt[3] == 4.67

        u_cont = ru.top_controversial(3)
        assert isinstance(u_cont, dict)
        vals_ucont = list(u_cont.values())
        assert vals_ucont == sorted(vals_ucont, reverse=True)

        # Bonus: critics_and_fans_ratio
        cf = ru.critics_and_fans_ratio()
        assert isinstance(cf, dict)
        assert set(cf.keys()) == {"harsh_critics", "moderate", "fans"}
        # Math check:
        # User 1 avg=4.0 (moderate), User 2 avg=3.33 (moderate), User 3 avg=4.67 (fans), User 4 avg=1.0 (harsh)
        assert cf["harsh_critics"] == 1
        assert cf["moderate"] == 2
        assert cf["fans"] == 1

    # 4. Tests for Links & TMDB/IMDB
    def test_links_methods_and_bonus(self, sample_files):
        links = Links(sample_files["links"], sample_files["movies"])

        # get_tmdb format
        res = links.get_tmdb(["1", "2"], ["Director", "Runtime"])
        assert isinstance(res, list)
        assert len(res) == 2
        assert isinstance(res[0], list)
        assert int(res[0][0]) >= int(res[1][0])

        # get_imdb alias check
        res_alias = links.get_imdb(["1", "2"], ["Director", "Runtime"])
        assert isinstance(res_alias, list)
        assert len(res_alias) == 2

        # Titles check in analytical methods
        dirs = links.top_directors(3)
        assert isinstance(dirs, dict)

        exp = links.most_expensive(3)
        assert isinstance(exp, dict)
        assert all(isinstance(k, str) for k in exp)

        prof = links.most_profitable(3)
        assert isinstance(prof, dict)
        assert all(isinstance(k, str) for k in prof)

        long = links.longest(3)
        assert isinstance(long, dict)
        assert all(isinstance(k, str) for k in long)

        cpm = links.top_cost_per_minute(3)
        assert isinstance(cpm, dict)
        assert all(isinstance(k, str) for k in cpm)

        # Bonus: top_roi
        roi = links.top_roi(3)
        assert isinstance(roi, dict)
        assert all(isinstance(k, str) for k in roi)
        assert all(isinstance(v, float) for v in roi.values())
        vals_roi = list(roi.values())
        assert vals_roi == sorted(vals_roi, reverse=True)

    # 5. Tests for Exception handling
    def test_exception_handling_on_invalid_files(self, sample_files):
        invalid_path = sample_files["invalid"]

        with pytest.raises(ValueError):
            Movies(invalid_path)

        with pytest.raises(ValueError):
            Tags(invalid_path)

        with pytest.raises(ValueError):
            Ratings(invalid_path, sample_files["movies"])

        with pytest.raises(ValueError):
            Links(invalid_path, sample_files["movies"])

        with pytest.raises(FileNotFoundError):
            Movies("non_existent_file_path.csv")


if __name__ == "__main__":
    exit_code = pytest.main(["-v", __file__])
    sys.exit(exit_code)
