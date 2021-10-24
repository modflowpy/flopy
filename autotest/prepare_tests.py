"""
Script to be used to download any required data prior to autotests
"""

from t503_test import download_mf6_examples


def test_mf6_download():
    """Download MODFLOW 6 examples in latest release"""
    download_mf6_examples(delete_existing=True)


if __name__ == "__main__":
    test_mf6_download()
