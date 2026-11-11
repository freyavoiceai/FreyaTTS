from freyatts.pipeline import expand_digits


def test_clock_minute_keeps_leading_zero():
    assert expand_digits("Saat 09:05'te") == "Saat dokuz sıfır beş'te"
    assert expand_digits("Saat 14:00'te") == "Saat on dört'te"
    assert expand_digits("Saat 14:45") == "Saat on dört kırk beş"


def test_decimal_comma():
    assert expand_digits("Faiz 3,05 oldu.") == "Faiz üç virgül sıfır beş oldu."
    assert expand_digits("Faiz 3,5 oldu.") == "Faiz üç virgül beş oldu."
    assert expand_digits("Bakiye 1.250,75") == "Bakiye bin iki yüz elli virgül yetmiş beş"


def test_percent():
    assert expand_digits("%15 indirim") == "yüzde on beş indirim"
    assert expand_digits("% 3,5 faiz") == "yüzde üç virgül beş faiz"


def test_unchanged():
    assert expand_digits("1, 2, 3") == "bir, iki, üç"
    assert expand_digits("Poliçe 4012345.") == "Poliçe dört sıfır bir iki üç dört beş."
    assert expand_digits("1.250 kişi") == "bin iki yüz elli kişi"
    assert expand_digits("%") == "%"
