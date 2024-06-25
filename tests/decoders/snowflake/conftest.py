import pytest

from localuf import Repetition
from localuf.decoders import Snowflake
from localuf.decoders.snowflake import _Node, _Edge, Snowflake
from localuf.type_aliases import Node


@pytest.fixture(
        name="rp_frugal",
        params=range(3, 11, 2),
        ids=lambda x: f"d{x}",
)
def _rp_frugal(request):
    return Repetition(
        request.param,
        noise='phenomenological',
        scheme='frugal',
    )


@pytest.fixture
def snowflake(rp_frugal: Repetition):
    return Snowflake(rp_frugal)


@pytest.fixture
def snowflake3():
    rp = Repetition(
        3,
        noise='phenomenological',
        scheme='frugal',
    )
    return Snowflake(rp)


@pytest.fixture
def sfn3(snowflake3: Snowflake):
    return snowflake3.NODES[0, 0]


@pytest.fixture
def fixture_test_INDEX_property():
    def f(obj: _Node | _Edge):
        assert type(obj.INDEX) is tuple
    return f


@pytest.fixture
def sfe3(snowflake3: Snowflake):
    return snowflake3.EDGES[(0, 0), (1, 0)]


@pytest.fixture
def syncing_flooding_objects(snowflake3: Snowflake) -> tuple[
    Snowflake, tuple[Node, Node, Node], tuple[_Node, _Node, _Node]
]:
    w, c, e = (-1, 0), (0, 0), (1, 0)
    west = snowflake3.NODES[w]
    center = snowflake3.NODES[c]
    east = snowflake3.NODES[e]
    return snowflake3, (w, c, e), (west, center, east)