import json
import pytest
from functools import partial
from operator import methodcaller
from src import todo
from src.utils.fn import compose


def test_get_projects(capsys: pytest.CaptureFixture):
    # Arrange
    todo.TODO_STATE_PATH.projects.unlink(missing_ok=True)

    # Act
    todo.main('get project'.split())

    # Assert
    stdout, _ = capsys.readouterr()
    expected_projects = sorted(
        [
            "Inbox",
            "Work",
            "Habits",
            "Someday",
            "Shopping list",
            "Projects",
        ]
    )
    split_lines = methodcaller('split', '\n')
    line2Project = methodcaller('removeprefix', '- ')
    to_projects = compose(
        partial(filter, None), partial(map, line2Project), split_lines
    )

    assert sorted([s for s in to_projects(stdout) if s]) == expected_projects
    with open(todo.TODO_STATE_PATH.projects) as f:
        projects = json.load(f)
        assert sorted([p['name'] for p in projects]) == expected_projects

def test_get_active_tasks():
    # Arrange
    todo.TODO_STATE_PATH.tasks.unlink(missing_ok=True)

    # Act
    todo.main('get task'.split())

    # Assert
    with open(todo.TODO_STATE_PATH.tasks) as f:
        tasks = json.load(f)
        assert tasks

def test_add_task():
    # -- Add
    # Arrange
    content = 'Buy Milk'
    cmd = ['add', 'task', '-c',content]

    # Act
    todo.main(cmd)

    # Assert
    assert any(t.content == content for t in todo.get_active_tasks())

    # -- Delete
    # Arrange
    index = 1
    cmd = ['delete', 'task', f'{index}']

    # Act
    todo.main(cmd)
    assert all(t.content != content for t in todo.get_active_tasks())


def test_update_task():
    # -- Add
    # Arrange
    og_content = 'Buy Milk'
    cmd = ['add', 'task', '-c',og_content]

    # Act
    todo.main(cmd)

    # Assert
    assert any(t.content == og_content for t in todo.get_active_tasks())

    # -- update
    # Arrange
    up_content = 'Buy Chocolate'
    cmd = ['update', 'task', '1', '-c', up_content]

    # Act
    todo.main(cmd)

    # Assert
    assert all(t.content != og_content for t in todo.get_active_tasks())
    assert any(t.content == up_content for t in todo.get_active_tasks())

    # -- Delete
    # Arrange
    index = 1
    cmd = ['delete', 'task', f'{index}']

    # Act
    todo.main(cmd)

    # Asset
    assert all(t.content != up_content for t in todo.get_active_tasks())


def test_close_task():
    # -- Add
    # Arrange
    og_content = 'Buy Milk'
    cmd = ['add', 'task', '-c',og_content]

    # Act
    todo.main(cmd)

    # Assert
    assert any(t.content == og_content for t in todo.get_active_tasks())

    # -- Close
    # Arrange
    index = 1
    cmd = ['close', 'task', f'{index}']

    # Act
    todo.main(cmd)

    # Asset
    assert all(t.content != og_content for t in todo.get_active_tasks())

def test_add_task_with_parent():
    # -- Add
    # Arrange
    og_content = 'Buy Milk'
    cmd = ['add', 'task', '-c',og_content]
    todo.main(cmd)
    next_content = 'Buy Chocolate'
    cmd = ['add', 'task', '-c', next_content, '--parent', '1']

    # Act
    todo.main(cmd)

    # Assert
    og_task = [t for t in todo.get_active_tasks() if t.content == og_content ][0]
    next_task = [t for t in todo.get_active_tasks() if t.content == next_content ][0]
    assert next_task.parent_id == og_task.id

    # -- cleanup
    # Arrange
    cmd = ['delete', 'task', '1', '2']

    # Act
    todo.main(cmd)

