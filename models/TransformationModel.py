class TransformationModel:
    def __init__(self, name, transformation):
        self.__name = name
        self.__transformation = transformation

    def get_name(self):
        return self.__name

    def get_transformation(self):
        return self.__transformation