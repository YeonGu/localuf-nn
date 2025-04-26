from abc import abstractmethod
from localuf._base_classes import Code, Decoder
from .nncore import NNCore
from ...type_aliases import Node


class NN(Decoder):

    def __init__(self, code: Code):
        super().__init__(code)
        print("Graph edges and their attributes:")
        for edge in code.GRAPH.edges(data=True):
            u, v, attrs = edge
            print(f"Edge ({u}, {v}): {attrs}")
        self._core = NNCore(code)

    def decode(self, syndrome: set[Node], **kwargs):
        print(f"Decoding syndrome: {syndrome}")
        self._core.decode(syndrome, self._core.STRATEGY.NEAREST)
        raise NotImplementedError

    def draw_decode(self, **kwargs_for_networkx_draw):
        raise NotImplementedError
