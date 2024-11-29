from src import ocd
from pathlib import Path
import pytest


@pytest.mark.parametrize(
    ('input_files', 'expected'),
    (
        pytest.param(
            [
                'screenshot_2024-06-02.png',
                'screenshot_2024-06-05.png',
                '2025-02-03.txt',
                '2025-09-10.txt',
            ],
            [
                '2024/jun/screenshot_2024-06-02.png',
                '2024/jun/screenshot_2024-06-05.png',
                '2025/feb/2025-02-03.txt',
                '2025/sep/2025-09-10.txt',
            ],
            id='Trivial Case',
        ),
        pytest.param(
            [
                '2024/jun/screenshot_2024-06-02.png',
                '2024/jun/screenshot_2024-06-05.png',
                '2025/feb/2025-02-03.txt',
                '2025/sep/2025-09-10.txt',
            ],
            [
                '2024/jun/screenshot_2024-06-02.png',
                '2024/jun/screenshot_2024-06-05.png',
                '2025/feb/2025-02-03.txt',
                '2025/sep/2025-09-10.txt',
            ],
            id='Already organized',
        ),
    ),
)
def test_organize_files(tmp_path: Path, input_files: list[str], expected: list[str]) -> None:
    # Arrange
    def str2path(file: str) -> Path:
        return tmp_path / file

    dirs = ['some_dir1', 'some_dir2']
    for dir in map(str2path, dirs):
        dir.mkdir(exist_ok=True)

    for file in map(str2path, input_files):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()

    # Act
    ocd.main([str(tmp_path)])

    # Assert
    pfiles, pexpected = map(str2path, input_files), map(str2path, expected)
    for in_file, out_file in zip(pfiles, pexpected):
        if in_file == out_file:
            assert out_file.is_file()
        else:
            assert not in_file.is_file()
            assert out_file.is_file()

    for dir in map(str2path, dirs):
        assert dir.is_dir()


