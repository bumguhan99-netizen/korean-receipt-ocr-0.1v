from metrics import measure, aggregate, distance


def main():
    assert distance('', 'abc') == 3
    assert distance('kitten', 'sitting') == 3
    assert measure('38,000', '38000')['char_edits'] == 1
    assert measure('a b', 'ab')['char_edits'] == 1
    assert measure('a b', 'ab')['word_edits'] == 2
    assert measure('A', 'a')['exact_match'] == 0
    assert measure(' 가 ', '가')['char_edits'] == 2
    assert measure('\u1100\u1161', '가')['exact_match'] == 1
    assert measure('12:30', '12:30')['numeric_symbol_only'] == 1
    assert measure('12원', '12원')['numeric_symbol_only'] == 0
    assert measure('A12', 'A12')['numeric_symbol_only'] == 0
    assert aggregate([measure('a', 'b'), measure('123456789', '123456789')])['cer_pct'] == 10
    assert aggregate([measure('a', 'aaaa')])['cer_pct'] == 300
    print('Metric checks passed: NFC, punctuation, whitespace, micro averaging, insertions.')


if __name__ == '__main__':
    main()
