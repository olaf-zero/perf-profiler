
class ProfilerError(Exception):
    pass

class SessionNotFoundError(ProfilerError, KeyError):
    pass
