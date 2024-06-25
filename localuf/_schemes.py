from abc import abstractmethod
import itertools
from typing import Literal

from localuf import constants
from localuf.noise import CodeCapacity
from localuf.type_aliases import Edge, Node
from localuf._base_classes import Scheme, Code, Decoder
from localuf._pairs import LogicalCounter, Pairs
from localuf._determinants import SpaceDeterminant, SpaceTimeDeterminant


class Batch(Scheme):
    """Batch decoding scheme.

    Extends `Scheme`.

    Overriden methods:
    * `run`
    * `sim_cycles_given_weight`
    * `get_logical_error`
    """

    @staticmethod
    def __str__() -> str:
        return 'batch'

    def __init__(self, code: Code, window_height: int):
        """Additional inputs: `window_height` the height of the batch."""
        super().__init__(code)
        self._WINDOW_HEIGHT = window_height
        self._DETERMINANT = SpaceDeterminant(code.D, code.LONG_AXIS)

    @property
    def WINDOW_HEIGHT(self): return self._WINDOW_HEIGHT

    def run(self, decoder: Decoder, p: float, n: int):
        # this assumes logical error count per batch << 1
        m = sum(self._sim_cycle_given_p(decoder, p) for _ in itertools.repeat(None, n))
        return (m, n if isinstance(self._CODE.NOISE, CodeCapacity)
            else self.WINDOW_HEIGHT * n / self._CODE.D)

    def get_logical_error(self, leftover):
        """Return logical error count parity in `leftover`."""
        ct: int = 0
        for u, _ in leftover:
            ct += (u[self._CODE.LONG_AXIS] == -1)
        return ct % 2

    def _sim_cycle_given_p(self, decoder: Decoder, p: float) -> int:
        """Simulate a decoding cycle given `p`.

        Input:
        * `decoder` the decoder.
        * `p` physical error probability.

        Output: `0` if success else `1`.
        """
        error = self._CODE.make_error(p)
        return self._sim_cycle_given_error(decoder, error)

    def _sim_cycle_given_error(self, decoder: Decoder, error: set[Edge]):
        """Simulate a decoding cycle given `error`.

        Input:
        * `error` the set of bitflipped edges.

        Output: `0` if success else `1`.
        """
        syndrome = self._CODE.get_syndrome(error)
        decoder.reset()
        decoder.decode(syndrome)
        leftover = error ^ decoder.correction
        return self.get_logical_error(leftover)

    def sim_cycles_given_weight(self, decoder, weight, n):
        m = 0
        for _ in itertools.repeat(None, n):
            error = self._CODE.NOISE.force_error(weight)
            m += self._sim_cycle_given_error(decoder, error)
        return m, n


class Global(Batch):
    """Global batch decoding scheme.

    Extends `Batch`.

    Additional attributes:
    * `pairs` a set of node pairs defining free anyon strings.
    Used to count logical error strings.

    Overriden methods:
    * `get_logical_error`
    * `run`
    """

    @staticmethod
    def __str__() -> str:
        return 'global batch'

    def __init__(self, code: Code, window_height: int):
        self.pairs = Pairs()
        super().__init__(code, window_height)

    def reset(self):
        """Factory reset."""
        self.pairs.reset()

    def run(self, decoder: Decoder, p: float, n: int):
        # `slenderness = n` assumes the following:
        assert self.WINDOW_HEIGHT == self._CODE.D * n
        self.reset()
        m = self._sim_cycle_given_p(decoder, p)
        return m, n

    def get_logical_error(self, leftover: set[Edge]):
        """Count logical errors in `leftover`."""
        for e in leftover:
            self.pairs.load(e)
        ct: int = 0
        visited: set[Node] = set()
        for u, v in self.pairs.dc.items():
            if u not in visited:
                pair_separation = abs(u[self._CODE.LONG_AXIS] - v[self._CODE.LONG_AXIS])
                if pair_separation == self._CODE.D:
                    ct += 1
                visited.add(v)
        return ct


class _Streaming(Scheme):
    """Abstract base class for stream decoding schemes.
    
    Extends `Scheme`.

    Additional attributes:
    * `_COMMIT_HEIGHT` the height of the commit region.
    * `_BUFFER_HEIGHT` the height of the buffer region.
    * `_COMMIT_EDGES` the edges of the commit region.
    * `_BUFFER_EDGES` the edges of the buffer region.
    * `pairs` a set of node pairs defining free anyon strings.
    Used to count logical error strings.
    * `_LOGICAL_COUNTER` for `get_logical_error`.

    Overriden methods:
    * `get_logical_error`
    """

    def __init__(
            self,
            code: Code,
            commit_height: int,
            buffer_height: int,
            commit_edges: tuple[Edge, ...],
    ):
        """Additional inputs:
        * `commit_height` the height of the commit region.
        * `buffer_height` the height of the buffer region.
        * `commit_edges` the edges in the commit region.
        """
        super().__init__(code)
        self._COMMIT_HEIGHT = commit_height
        self._BUFFER_HEIGHT = buffer_height
        self._COMMIT_EDGES = commit_edges
        self._BUFFER_EDGES = tuple(set(code.EDGES) - set(commit_edges))
        self._DETERMINANT = SpaceTimeDeterminant(
            d=code.D,
            long_axis=code.LONG_AXIS,
            time_axis=code.TIME_AXIS,
            window_height=self.WINDOW_HEIGHT,
        )
        self.pairs = Pairs()
        self._LOGICAL_COUNTER = LogicalCounter(
            d=code.D,
            commit_height=commit_height,
            long_axis=code.LONG_AXIS,
            time_axis=code.TIME_AXIS,
        )

    @property
    def WINDOW_HEIGHT(self): return self._COMMIT_HEIGHT + self._BUFFER_HEIGHT

    @abstractmethod
    def reset(self):
        """Factory reset."""
        self.pairs.reset()

    def get_logical_error(self):
        """Count logical errors completed in current commit.

        Output:
        The number of logical errors,
        i.e. paths between opposite boundaries,
        completed by the leftover in the current commit region.

        Side effect:
        Update `self.pairs` with error strings ending
        at the temporal boundary of the commit region.
        """
        ct, self.pairs = self._LOGICAL_COUNTER.count(self.pairs)
        return ct
    
    def sim_cycles_given_weight(self, decoder, weight, n):
        raise NotImplementedError("Not implemented for streaming schemes.")


class Forward(_Streaming):
    """Forward decoding scheme. Also known as overlapping recovery method.

    Extends `_Streaming`.

    Additional attributes:
    * `history` a list of tuples (error, leftover, artificial defects) for each cycle.

    Overriden methods:
    * `reset`
    * `get_logical_error`
    * `run`
    """

    @staticmethod
    def __str__() -> str:
        return 'forward'

    def reset(self):
        super().reset()
        try: del self.history
        except: pass

    def _make_error(self, buffer_leftover: set[Edge], p: float):
        """Lower `buffer_leftover` by commit height
        and sample edges from freshly discovered region with probability `p`.

        Input:
        * `buffer_leftover` the current error in the buffer region.
        * `p` probability for an edge to bitflip.

        Output: The set of bitflipped edges.
        """
        # lower `buffer_leftover` by commit height
        seen: set[Edge] = set()
        for e in buffer_leftover:
            seen.add(self._CODE.raise_edge(e, -self._COMMIT_HEIGHT))
        # populate freshly discovered region with new errors
        unseen = self._CODE.make_error(p)
        return seen | unseen

    def _get_leftover(self, error: set[Edge], correction: set[Edge]):
        """Sequentially compose `error` and commit region of `correction`.

        Input:
        * `error` the set of bitflipped edges in the window.
        * `correction` the decoder output for the whole window.

        Output:
        * `commit_leftover` the leftover,
        i.e. sequential composition of `error` and `correction`,
        in the commit region.
        * `buffer_leftover` the part of `error` in the buffer region.
        """
        commit_leftover = error.intersection(self._COMMIT_EDGES) ^ correction.intersection(self._COMMIT_EDGES)
        buffer_leftover = error.intersection(self._BUFFER_EDGES)
        return commit_leftover, buffer_leftover

    def get_logical_error(self, commit_leftover: set[Edge]):
        """Count logical errors completed in current commit.

        Additional input over `_Streaming.get_logical_error`:
        `commit_leftover` the leftover in the commit region.
        """
        for e in commit_leftover:
            self.pairs.load(e)
        return super().get_logical_error()

    def _get_syndrome(
            self,
            commit_leftover: set[Edge],
            error: set[Edge],
    ):
        """Get syndrome of `error` accounting for artificial defects due to previous commit."""
        artificial_defects = self._get_artificial_defects(commit_leftover)
        return self._CODE.get_syndrome(error) ^ artificial_defects

    def _get_artificial_defects(self, commit_leftover: set[Edge]):
        """Get artificial defects due to `commit_leftover`.

        See Skoric et al. [arXiv:2209.08552v2, Section I B]
        for artificial defect definition.
        """
        commit_syndrome = self._CODE.get_syndrome(commit_leftover)
        return {self._CODE.raise_node(v, -self._COMMIT_HEIGHT) for v in commit_syndrome
                if v[self._CODE.TIME_AXIS] == self._COMMIT_HEIGHT}

    def _make_error_in_buffer_region(self, p: float):
        """Sample edges from buffer region.
        
        Input:
        `p` characteristic probability if circuit-level noise;
        else, bitflip probability.

        Output:
        The set of bitflipped edges in the buffer region.
        Each edge bitflips with
        probability defined by its multiplicity if circuit-level noise; else,
        probability `p`.

        TODO: test this method.
        """
        if self._COMMIT_HEIGHT >= self._BUFFER_HEIGHT:
            error = self._CODE.make_error(p)
        else:
            rep_count = self._BUFFER_HEIGHT // self._COMMIT_HEIGHT + 1
            error: set[Edge] = set()
            for _ in itertools.repeat(None, rep_count):
                error = self._make_error(error, p)
        return error & set(self._BUFFER_EDGES)
    
    def run(
            self,
            decoder: Decoder,
            p: float,
            n: int,
            draw=False,
            log_history=False,
            **kwargs,
    ):
        log_history |= draw
        self.reset()
        m = 0
        commit_leftover: set[Edge] = set()
        buffer_leftover = self._make_error_in_buffer_region(p)
        # `cleanse_count` additional decoding cycles ensures window is free of defects
        cleanse_count = self.WINDOW_HEIGHT // self._COMMIT_HEIGHT
        probs = itertools.chain(itertools.repeat(p, n-1), itertools.repeat(0, cleanse_count))
        if log_history:
            self.history: list[tuple[set[Edge], set[Edge], set[Node]]] = []
            for prob in probs:
                error = self._make_error(buffer_leftover, prob)
                artificial_defects = self._get_artificial_defects(commit_leftover)
                syndrome = self._CODE.get_syndrome(error) ^ artificial_defects
                decoder.reset()
                decoder.decode(syndrome)
                commit_leftover, buffer_leftover = self._get_leftover(error, decoder.correction)
                m += self.get_logical_error(commit_leftover)
                self.history.append((error, commit_leftover | buffer_leftover, artificial_defects))
            if draw:
                self._draw_run(**kwargs)
        else:
            for prob in probs:
                error = self._make_error(buffer_leftover, prob)
                syndrome = self._get_syndrome(commit_leftover, error)
                decoder.reset()
                decoder.decode(syndrome)
                commit_leftover, buffer_leftover = self._get_leftover(error, decoder.correction)
                m += self.get_logical_error(commit_leftover)
        return m, (self._BUFFER_HEIGHT + (n-1)*self._COMMIT_HEIGHT) / self._CODE.D

    def _draw_run(
        self,
        fig_width: float | None = None,
        x_offset=constants.STREAM_X_OFFSET,
        subplot_hspace=-0.1,
    ):
        """Draw the history of `self.run`."""
        import matplotlib.pyplot as plt
        column_count, row_count = len(self.history), 2
        if fig_width is None:
            fig_width = 1.5 if self._CODE.DIMENSION==2 else self._CODE.D*3/5
        plt.figure(figsize=(
            fig_width * column_count,
            fig_width * row_count * self.WINDOW_HEIGHT/self._CODE.D
        ))
        for k, (error, leftover, artificial_defects) in enumerate(self.history, start=1):
            for l, edges in enumerate((error, leftover)):
                plt.subplot(row_count, column_count, column_count*l+k)
                self._CODE.draw(
                    edges,
                    syndrome=self._CODE.get_syndrome(edges) ^ artificial_defects,
                    x_offset=x_offset,
                    with_labels=False,
                )
        plt.tight_layout()
        plt.subplots_adjust(hspace=subplot_hspace)


class Frugal(_Streaming):
    """Frugal decoding scheme.

    Extends `_Streaming`.

    Additional attributes:
    * `error` a set of edges.
    * `step_counts` a list in which each entry is the decoder timestep count of `d` decoding cycles.
    Populated when `run` is called.
    * `_temporal_boundary_syndrome` the set of defects in the temporal boundary of the viewing window.

    Overriden methods:
    * `reset`
    * `run`
    """

    @staticmethod
    def __str__() -> str:
        return 'frugal'

    def __init__( self, code, commit_height, buffer_height, commit_edges):
        super().__init__(code, commit_height, buffer_height, commit_edges)
        self.error: set[Edge] = set()
        self.step_counts: list[int] = []
        self._temporal_boundary_syndrome: set[Node] = set()

    def reset(self):
        super().reset()
        self.error.clear()
        self.step_counts.clear()
        self._temporal_boundary_syndrome.clear()

    def advance(self, prob: float, decoder: Decoder, **kwargs) -> int:
        """Advance 1 decoding cycle.

        Input:
        * `prob` instantaneous physical error probability.
        * `decoder` the frugal-compatible decoder to use.
        * `log_history, time_only` arguments of `decoder.decode`.

        Output: number of decoder timesteps to complete decoding cycle.
        """
        self._raise_window()
        error = self._CODE.make_error(prob)
        syndrome = self._load(error)
        return decoder.decode(syndrome, **kwargs) # type: ignore

    def _raise_window(self):
        """Raise window by `self._COMMIT_HEIGHT` layers.

        TODO: store `lowness, next_edge` as attributes of each edge.
        """
        next_error: set[Edge] = set()
        for e in self.error:
            lowness = sum(v[self._CODE.TIME_AXIS] == 0 for v in e)
            if lowness == 0:
                next_edge = self._CODE.raise_edge(e, delta_t=-self._COMMIT_HEIGHT)
                next_error.add(next_edge)
            else:
                self.pairs.load(e)
        self.error = next_error

    def _load(self, error: set[Edge]):
        """Load incremental `error`.

        Input: `error` the incremental error, which should never intersect `self.error`.
        Output: `syndrome` the incremental syndrome due to `error`.
        Side effect: Update `self.error` and `self.temporal_boundary_syndrome`.
        """
        self.error |= error
        syndrome = {self._CODE.raise_node(v, delta_t=-self._COMMIT_HEIGHT)
            for v in self._temporal_boundary_syndrome}
        self._temporal_boundary_syndrome.clear()
        for e in error:
            for v in e:
                if not self.is_boundary(v):
                    syndrome.symmetric_difference_update({v})
                elif v[self._CODE.TIME_AXIS] == self.WINDOW_HEIGHT:  # in temporal boundary
                    self._temporal_boundary_syndrome.symmetric_difference_update({v})
        return syndrome

    def run(
            self,
            decoder: Decoder,
            p: float,
            n: int,
            draw: Literal[False, 'fine', 'coarse'] = False,
            log_history: Literal[False, 'fine', 'coarse'] = False,
            time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
            **kwargs,
    ):
        """Simulate `n*d` decoding cycles given `p`.

        Input:
        * `decoder` the decoder.
        * `p` physical error probability.
        * positive integer `n` is slenderness :=
        (measurement round count + 1) / (code distance).
        * `draw` whether to draw.
        * `log_history` whether to populate `history` attribute.
        * `time_only` whether runtime includes a timestep
        for each drop, each grow, and each merging step ('all');
        each merging step only ('merging');
        or each unrooting step only ('unrooting').
        Note: changing from 'merging' to 'all' simply increases each step count by 2d
        [for a total step count increase of 2d(n-1)].
        This can be done post-run via `from_merging_to_all`.
        * `kwargs` passed to `decoder.draw_decode`
        e.g. `margins=(0.1, 0.1)`.

        Output: tuple of (failure count, `n`).

        Side effect: Populate `self.step_counts` with `n-1` entries,
        each being the step count of `d` decoding cycles.
        """
        d = self._CODE.D
        self.reset()
        decoder.reset()
        if draw:
            log_history = draw
        if log_history:
            decoder.init_history() # type: ignore
        m = 0
        for prob, advance_count, time in itertools.chain(
            ((p, d, False),),
            itertools.repeat((p, d, True), n-1),
            ((0, 2*self.WINDOW_HEIGHT, False),),
        ):
            step_count = 0
            for _ in itertools.repeat(None, advance_count):
                step_count += self.advance(
                    prob,
                    decoder,
                    log_history=log_history,
                    time_only=time_only,
                )
                m += self.get_logical_error()
            if time:
                self.step_counts.append(step_count)
        if draw:
            decoder.draw_decode(**kwargs)
        return m, n