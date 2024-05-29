from utils.utils import get_char_re_expression


class CharsToRemove(object):
    list_chars_to_remove = ['#', '@', '\'', '´', "º", "ª", "°", "\"", "{", "}", "<", ">", ";", "=", "_", "-", "`", "~",
                            "%",
                            "/",
                            ":", "*",
                            "(", ")", "|", "\\n", "€", "\\", "&", "[", "]", ",", ".", "?", "¿", "!", "¡"]
    re_list_chars_to_remove = get_char_re_expression(list_chars_to_remove)
