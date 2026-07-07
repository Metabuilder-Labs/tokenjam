from tokenjam.utils.formatting import format_cost, format_tokens, severity_colour


class TestFormatCost:
    def test_small_cost(self):
        assert format_cost(0.0001) == "$0.00"

    def test_normal_cost(self):
        assert format_cost(0.034) == "$0.03"

    def test_zero_cost(self):
        assert format_cost(0.0) == "$0.00"

    def test_threshold_boundary(self):
        assert format_cost(0.001) == "$0.00"

    def test_sub_hundred_cost_keeps_two_decimals(self):
        assert format_cost(12.50) == "$12.50"

    def test_exactly_one_hundred_uses_whole_dollars(self):
        assert format_cost(100.0) == "$100"

    def test_large_cost_rounds_to_whole_dollars_with_separators(self):
        assert format_cost(1042.4825) == "$1,042"

    def test_very_large_cost_uses_thousands_separators(self):
        assert format_cost(29488.0100) == "$29,488"


class TestFormatTokens:
    def test_small_number(self):
        assert format_tokens(500) == "500"

    def test_thousands(self):
        assert format_tokens(1000) == "1.0k"

    def test_thousands_with_fraction(self):
        assert format_tokens(12447) == "12.4k"

    def test_millions(self):
        assert format_tokens(1_000_000) == "1.0M"

    def test_millions_with_fraction(self):
        assert format_tokens(2_500_000) == "2.5M"

    def test_zero(self):
        assert format_tokens(0) == "0"

    def test_just_below_thousand(self):
        assert format_tokens(999) == "999"


class TestSeverityColour:
    def test_critical(self):
        assert severity_colour("critical") == "red"

    def test_warning(self):
        assert severity_colour("warning") == "yellow"

    def test_info(self):
        assert severity_colour("info") == "blue"

    def test_unknown(self):
        assert severity_colour("unknown") == "white"
