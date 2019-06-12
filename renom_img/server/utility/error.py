class ReNomIMGError(Exception):
    code = "IMG0000"


class ReNomIMGUnknownError(ReNomIMGError):
    code = "IMG9999"
