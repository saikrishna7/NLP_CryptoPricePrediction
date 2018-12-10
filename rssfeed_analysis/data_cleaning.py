from data_initializer import DataInitializer

class DataCleaning(DataInitializer):
    def __init__(self, previous, is_testing):
        self.processed_data = previous.processed_data
        self.is_testing = is_testing

    def cleanup(self, cleanuper):
        t = self.processed_data
        for cleanup_method in cleanuper.iterate():
            if not self.is_testing:
                t = cleanup_method(t)
            else:
                if cleanup_method.__name__ != "remove_na":
                    t = cleanup_method(t)

        self.processed_data = t