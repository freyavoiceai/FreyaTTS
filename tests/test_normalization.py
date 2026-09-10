import unittest

from freyatts.pipeline import normalize


class NormalizationTests(unittest.TestCase):
    def test_preserves_leading_zeros(self):
        cases = {
            "0047": "sıfır sıfır dört yedi",
            "0850": "sıfır sekiz beş sıfır",
            "007": "sıfır sıfır yedi",
            "00": "sıfır sıfır",
            "01": "sıfır bir",
            "000047": "sıfır sıfır sıfır sıfır dört yedi",
            "Kartınızın son dört hanesi 0047.":
                "Kartınızın son dört hanesi sıfır sıfır dört yedi.",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)

    def test_zero_led_decimal_fractions_keep_their_value(self):
        # The fraction after a decimal comma is its own digit run, so the
        # leading-zero rule applies there too: 3,05 and 3,5 no longer collapse
        # to the same reading.
        cases = {
            "3,05": "üç,sıfır beş",
            "3,5": "üç,beş",
            "10,00": "on,sıfır sıfır",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)

    def test_existing_number_rules(self):
        cases = {
            "0": "sıfır",
            "47": "kırk yedi",
            "850": "sekiz yüz elli",
            "99999": "doksan dokuz bin dokuz yüz doksan dokuz",
            "123456": "bir iki üç dört beş altı",
            "1.250": "bin iki yüz elli",
            "08:05": "sekiz beş",
            "08:00": "sekiz",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)


if __name__ == "__main__":
    unittest.main()
