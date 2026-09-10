import unittest

from freyatts.pipeline import normalize, spell_number


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

    def test_dates(self):
        cases = {
            "Randevunuz 12.03.2026 tarihinde.":
                "Randevunuz on iki Mart iki bin yirmi altı tarihinde.",
            "01.03.2026": "bir Mart iki bin yirmi altı",
            "1.2.2026": "bir Şubat iki bin yirmi altı",
            "29.02.2024": "yirmi dokuz Şubat iki bin yirmi dört",
            "31.12.2026": "otuz bir Aralık iki bin yirmi altı",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)

    def test_invalid_dates_keep_the_integer_reading(self):
        for text in ["29.02.2025", "31.04.2026", "00.03.2026", "12.13.2026"]:
            with self.subTest(text=text):
                self.assertEqual(normalize(text), spell_number(int(text.replace(".", ""))))

    def test_lira_amounts(self):
        cases = {
            "Tutar 10,00 TL.": "Tutar on lira.",
            "12.450,75 TL": "on iki bin dört yüz elli lira yetmiş beş kuruş",
            "Borcunuz 1.250,00 ₺": "Borcunuz bin iki yüz elli lira",
            "₺1.250,05": "bin iki yüz elli lira beş kuruş",
            "3,5 TL": "üç lira elli kuruş",
            "0,05 TL": "beş kuruş",
            "250000 TL": "iki yüz elli bin lira",
            "10tl": "on lira",
            "10 TLX": "on TLX",
            "100 TL'ye kadar": "yüz TL'ye kadar",
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
            "1.234.567": "bir milyon iki yüz otuz dört bin beş yüz altmış yedi",
            "08:05": "sekiz beş",
            "08:00": "sekiz",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)


if __name__ == "__main__":
    unittest.main()
