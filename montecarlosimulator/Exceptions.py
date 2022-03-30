class SimulationFailureError(RuntimeError):
    def __init__(self, data, message=''):
        self.data = data
        message = message + '\nSimulation failed using data={!r}'
        super().__init__(message.format(self.data))
