class TextSearchPyError(Exception):
    """
    General textsearchpy exception
    """

    pass


class IndexingError(TextSearchPyError):
    """
    Error during appending of text to index
    """

    pass


class QueryParseError(TextSearchPyError):
    """
    Error parsing query string into valid Query object
    """

    pass
