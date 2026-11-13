
class ProfilerError(Exception):
    pass

class SessionNotFoundError(ProfilerError, KeyError):
    pass

class ThemeError(ProfilerError, Exception):
    pass

class ColorNotFoundError(ThemeError, ValueError):
    pass