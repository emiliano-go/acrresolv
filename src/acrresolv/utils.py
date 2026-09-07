from cleantext import clean_words


def clean_origin(origin: str, remove_stopwords: bool = True) -> list[str]:
    words = clean_words(
        origin,
        clean_all=False,
        numbers=False,
        stp_lang="english",
        stopwords=remove_stopwords,
    )
    return [w for w in words if w]


def get_initials(words: list[str]) -> list[str]:
    return [w[0].upper() for w in words if w and w[0].isalpha()]


def build_acro(initials: list[str]) -> str:
    return "".join(initials)


def rule_based(text: str, remove_stopwords: bool) -> str:
    words = clean_origin(text, remove_stopwords)
    initials = get_initials(words)
    return build_acro(initials)


def is_confident(acro: str) -> bool:
    return 2 <= len(acro) <= 6
