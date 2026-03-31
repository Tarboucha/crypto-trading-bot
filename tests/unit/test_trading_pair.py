"""Tests for TradingPair — parsing, exchange IDs, validation."""
import pytest
from shared.pairs import TradingPair, parse_pair, parse_pairs, validate_pairs


class TestParsing:

    @pytest.mark.parametrize("input_str,expected_base,expected_quote,expected_market", [
        ("ETH", "ETH", "USDT", "swap"),
        ("ETH/USDT", "ETH", "USDT", "swap"),
        ("ETH/USDT:USDT", "ETH", "USDT", "swap"),
        ("ETH/BTC", "ETH", "BTC", "swap"),
        ("BTC/EUR", "BTC", "EUR", "swap"),
        ("sol", "SOL", "USDT", "swap"),  # lowercase
    ])
    def test_parse_futures(self, input_str, expected_base, expected_quote, expected_market):
        pair = parse_pair(input_str, trading_mode="futures")
        assert pair.base == expected_base
        assert pair.quote == expected_quote
        assert pair.market_type == expected_market

    def test_parse_spot(self):
        pair = parse_pair("ETH/USDT", trading_mode="spot")
        assert pair.market_type == "spot"
        assert pair.settle is None

    def test_parse_futures_has_settle(self):
        pair = parse_pair("ETH/USDT", trading_mode="futures")
        assert pair.settle == "USDT"

    def test_parse_explicit_settle(self):
        pair = parse_pair("ETH/USDT:ETH", trading_mode="futures")
        assert pair.settle == "ETH"

    def test_parse_multiple(self):
        pairs = parse_pairs(["ETH/USDT", "BTC/USDT", "SOL/ETH"])
        assert len(pairs) == 3
        assert pairs[0].base == "ETH"
        assert pairs[2].base == "SOL"
        assert pairs[2].quote == "ETH"


class TestSymbol:

    def test_symbol_spot(self):
        pair = parse_pair("ETH/USDT", trading_mode="spot")
        assert pair.symbol == "ETH/USDT"

    def test_symbol_perp(self):
        pair = parse_pair("ETH/USDT", trading_mode="futures")
        assert pair.symbol == "ETH/USDT:USDT"

    def test_short_name_usdt(self):
        pair = parse_pair("ETH/USDT")
        assert pair.short_name == "ETH"

    def test_short_name_non_usdt(self):
        pair = parse_pair("ETH/BTC")
        assert pair.short_name == "ETH/BTC"


class TestExchangeId:

    @pytest.mark.parametrize("exchange,expected", [
        ("hyperliquid", "ETH"),
        ("binance", "ETHUSDT"),
    ])
    def test_exchange_id(self, exchange, expected):
        pair = parse_pair("ETH/USDT")
        assert pair.exchange_id(exchange) == expected

    def test_csv_symbol(self):
        pair = parse_pair("ETH/USDT")
        assert pair.csv_symbol == "ETHUSDT"

    def test_unknown_exchange_returns_symbol(self):
        pair = parse_pair("ETH/USDT")
        result = pair.exchange_id("unknown_exchange")
        assert result == pair.symbol


class TestValidation:

    def test_valid_pairs(self):
        pairs = parse_pairs(["ETH/USDT", "BTC/USDT"])
        result = validate_pairs(pairs)
        assert len(result) == 2

    def test_invalid_base(self):
        pairs = [parse_pair("FAKE/USDT")]
        with pytest.raises(ValueError, match="Unsupported base"):
            validate_pairs(pairs)

    def test_invalid_quote(self):
        pairs = [parse_pair("ETH/FAKE")]
        with pytest.raises(ValueError, match="Unsupported quote"):
            validate_pairs(pairs)


class TestImmutability:

    def test_frozen(self):
        pair = parse_pair("ETH/USDT")
        with pytest.raises(AttributeError):
            pair.base = "BTC"
