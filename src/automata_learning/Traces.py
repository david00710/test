class Traces:
    def __init__(self, positive = set(), negative = set()):
        self.positive = positive
        self.negative = negative

    """
     IG: at the moment we are adding a trace only if it ends up in an event.
     should we be more restrictive, e.g. consider xxx, the same as xxxxxxxxxx (where x is an empty event '')
     recent suggestion (from the meeting): ignore empty events altogether and don't consider them as events at all (neither for
     execution, nor for learning)
     """
    def _should_add(self, trace, i):
        prefixTrace = trace[:i]
        if not prefixTrace[-1] == '':
            return True
        else:
            return False

    def _get_prefixes(self, trace, up_to_limit = None):
        if up_to_limit is None:
            up_to_limit = len(trace)
        all_prefixes = set()
        for i in range(1, up_to_limit+1):
            if self._should_add(trace, i):
                all_prefixes.add(trace[:i])
        return all_prefixes


    """
    when adding a trace, it additionally adds all prefixes as negative traces
    """
    def add_trace(self, trace, reward):
        trace = tuple(trace)
        if reward > 0:
            self.positive.add(trace)
            # | is a set union operator
            self.negative |= self._get_prefixes(trace, len(trace)-1)

        else:
            self.negative |= self._get_prefixes(trace)

    def __repr__(self):
        return repr(self.positive) + "\n\n" + repr(self.negative)