class Graph:
    def __init__(self, tempo, performance, accuracy, validation_tempo, validation_performance, validation_accuracy,
                 fine_tuning=[]):
        self.tempo = tempo
        self.performance = performance
        self.accuracy = accuracy
        self.validation_tempo = tempo
        self.validation_performance = validation_performance
        self.validation_accuracy = validation_accuracy
        self.fine_tuning = fine_tuning
