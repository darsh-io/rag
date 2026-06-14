"""Class member (teacher/student) assignment tests."""
import pytest


@pytest.fixture
async def class_with_members(client, admin_headers):
    import uuid
    uid = uuid.uuid4().hex[:8]
    cls_r = await client.post("/classes", headers=admin_headers, json={"name": f"Member Class {uid}"})
    class_id = cls_r.json()["id"]

    t_r = await client.post("/users", headers=admin_headers, json={
        "username": f"teacher_{uid}", "password": "teacher123", "role": "teacher"
    })
    teacher_id = t_r.json()["id"]

    s_r = await client.post("/users", headers=admin_headers, json={
        "username": f"student_{uid}", "password": "student123", "role": "student"
    })
    student_id = s_r.json()["id"]

    return class_id, teacher_id, student_id


@pytest.mark.asyncio
async def test_assign_and_list_teacher(client, admin_headers, class_with_members):
    class_id, teacher_id, _ = class_with_members
    r = await client.post(
        f"/classes/{class_id}/teachers",
        headers=admin_headers,
        json={"user_id": teacher_id},
    )
    assert r.status_code == 200

    r2 = await client.get(f"/classes/{class_id}/teachers", headers=admin_headers)
    assert r2.status_code == 200
    assert any(t["id"] == teacher_id for t in r2.json())


@pytest.mark.asyncio
async def test_remove_teacher(client, admin_headers, class_with_members):
    import uuid
    class_id, teacher_id, _ = class_with_members
    # Add a second teacher so removing the first isn't the "last teacher" case
    uid2 = uuid.uuid4().hex[:8]
    t2 = await client.post("/users", headers=admin_headers, json={
        "username": f"teacher2_{uid2}", "password": "teacher123", "role": "teacher"
    })
    teacher2_id = t2.json()["id"]
    await client.post(f"/classes/{class_id}/teachers", headers=admin_headers, json={"user_id": teacher_id})
    await client.post(f"/classes/{class_id}/teachers", headers=admin_headers, json={"user_id": teacher2_id})
    r = await client.delete(f"/classes/{class_id}/teachers/{teacher_id}", headers=admin_headers)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_enroll_and_list_student(client, admin_headers, class_with_members):
    class_id, _, student_id = class_with_members
    r = await client.post(
        f"/classes/{class_id}/students",
        headers=admin_headers,
        json={"user_id": student_id},
    )
    assert r.status_code == 200

    r2 = await client.get(f"/classes/{class_id}/students", headers=admin_headers)
    assert r2.status_code == 200
    assert any(s["id"] == student_id for s in r2.json())


@pytest.mark.asyncio
async def test_unenroll_student(client, admin_headers, class_with_members):
    class_id, _, student_id = class_with_members
    await client.post(f"/classes/{class_id}/students", headers=admin_headers, json={"user_id": student_id})
    r = await client.delete(f"/classes/{class_id}/students/{student_id}", headers=admin_headers)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_update_class(client, admin_headers):
    cls_r = await client.post("/classes", headers=admin_headers, json={"name": "Old Name"})
    class_id = cls_r.json()["id"]
    r = await client.patch(f"/classes/{class_id}", headers=admin_headers, json={"name": "New Name"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_my_classes(client, admin_headers):
    r = await client.get("/classes/me", headers=admin_headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)
