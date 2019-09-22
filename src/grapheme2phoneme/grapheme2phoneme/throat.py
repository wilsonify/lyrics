import re
from grapheme2phoneme.phonemes import (
    phonemes_dictionary,
    rules_English_to_phonemes_special_symbols,
    rules_English_to_phonemes
)


def ensure_text_alphanumeric(text=None):
    characters_pass = " 0123456789abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXYZ"
    characters_pass_set = set(characters_pass)
    return "".join(filter(characters_pass_set.__contains__, text)).strip("\n")


def match_and_replace(
        text=None,
        rule=None,
        phoneme=None
):
    """
    Replace found text from a single rule.
    """
    # Find all rule matches.
    matches = [
        (match.start(), match.end()) for match in re.finditer(rule, text)
    ]
    # Start from behind, so replace in-place.
    matches.reverse()
    # Convert to characters because strings are immutable.
    characters = list(text)
    for start, end in matches:
        characters[start:end] = phoneme
    # Convert back to string.
    return "".join(characters)


def make_regex_fragment_from_rules_English_to_phonemes_special_symbols(
        rule_pattern=None
):
    regex = r""
    for character in rule_pattern:
        regex += rules_English_to_phonemes_special_symbols.get(
            character, character
        )
    return regex


def make_rule_regex(
        rule_text=None
):
    character_string, left_context, right_context, phoneme = rule_text.split("/")
    rule = r""
    if left_context:
        # Use a non-capturing group to match the left context.
        rule += \
            r"(?:" + \
            make_regex_fragment_from_rules_English_to_phonemes_special_symbols(
                rule_pattern=left_context
            ) + \
            ")"
    # Create a capturing group for the character string.
    rule += r"(?P<found>" + character_string + ")"
    if right_context:
        # Add a lookahead pattern.
        rule += \
            r"(?=" + \
            make_regex_fragment_from_rules_English_to_phonemes_special_symbols(
                rule_pattern=right_context
            ) + \
            ")"
    # Return a tuple containing the regex created from the rule, a lower-case
    # representation of the phonemes between dashes and the original rule.
    return rule, "-{phoneme}-".format(phoneme=phoneme.lower()), rule_text


rules_English_to_phonemes_regex = [
    make_rule_regex(rule_text=rule) for rule in rules_English_to_phonemes
]


def text_to_phonemes(
        text=None,
        explain=False,
        _phonemes_dictionary=phonemes_dictionary
):
    """
    Extract phonemes from words.
    """
    if explain:
        print("\ntranslation printout:")
        print("text: {text}".format(text=text))

    text = ensure_text_alphanumeric(text=text)

    # Add space around words for compatibility with rules containing spaces.
    result = " {text} ".format(text=text.upper())
    step = 0

    # Iterate over all the interesting tuples.
    for rule, phoneme, rule_text in rules_English_to_phonemes_regex:
        # For each rule, 'tmp' is the string in which all matches for 'rule'
        # have been replaced by 'phoneme'.
        tmp = match_and_replace(
            text=result,
            rule=rule,
            phoneme=phoneme
        )
        if explain and tmp != result:
            step += 1
            message = \
                "step {step}: {result} ---> {tmp} [rule: {rule_text} ({rule})]"
            print(message.format(
                step=step,
                result=result,
                tmp=tmp,
                rule_text=rule_text,
                rule=rule
            ))

        result = tmp

    # remove artifacts
    result_artifacts_removed = result.replace(
        "- -",
        " "
    ).replace(
        "--",
        "-"
    ).strip(
        " "
    ).strip(
        "-"
    ).replace(
        "--",
        "-"
    )

    if explain:
        print("result: {result}\n".format(result=result_artifacts_removed))
    # make uppercase
    result_uppercase = result_artifacts_removed.upper()
    # remove junk
    acceptable_phonemes = _phonemes_dictionary.keys()
    result_cleaning = []
    for word in result_uppercase.split(" "):
        tmp_word = []
        for word_phoneme in word.split("-"):
            if word_phoneme in acceptable_phonemes:
                tmp_word.append(word_phoneme)
            if word_phoneme == "I":
                tmp_word.append("AH-EE")
            if word_phoneme == "EEH":
                tmp_word.append("EH")
        result_cleaning.append("-".join(tmp_word))

    result = " ".join(result_cleaning).strip()
    return result
