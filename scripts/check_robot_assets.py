"""Check bundled URDF, mesh, source scene and runtime assets using only the standard library."""
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]


def relative_file(base, name, root):
    path = (base/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or non-portable model dependency: '+name)
    return path


def check(project=PROJECT):
    robot = project/'robots/dogleg'
    manifest = json.loads((project/'docs/robot_assets_manifest.json').read_text(encoding='utf-8'))
    entries = manifest['files']+manifest['runtime_assets']
    for item in entries:
        path = relative_file(project, item['path'], project)
        raw = path.read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
    trees = []
    for path in sorted((robot/'urdf').glob('*.urdf')):
        xml = ET.parse(path).getroot()
        links = xml.findall('link')
        joints = xml.findall('joint')
        names = {item.get('name') for item in links}
        assert len(links) == len(names) == 19 and len(joints) == 18
        assert len({item.get('name') for item in joints}) == 18
        parents = {}
        for joint in joints:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            assert parent in names and child in names and child not in parents
            parents[child] = parent
        roots = names-set(parents)
        assert roots == {'base_link'}
        for name in names:
            seen = set()
            while name in parents:
                assert name not in seen, 'URDF cycle'
                seen.add(name)
                name = parents[name]
            assert name == 'base_link'
        total = 0.
        for link in links:
            inertia = link.find('inertial')
            mass = float(inertia.find('mass').get('value'))
            a = {k: float(v) for k, v in inertia.find('inertia').attrib.items()}
            assert math.isfinite(mass) and mass > 0 and all(math.isfinite(v) for v in a.values())
            determinant = a['ixx']*a['iyy']*a['izz']+2*a['ixy']*a['ixz']*a['iyz']-a['ixx']*a['iyz']**2-a['iyy']*a['ixz']**2-a['izz']*a['ixy']**2
            assert a['ixx'] > 0 and a['ixx']*a['iyy']-a['ixy']**2 > 0 and determinant > 0
            total += mass
        meshes = {relative_file(path.parent, item.get('filename'), robot) for item in xml.findall('.//mesh')}
        assert len(meshes) == 19
        assert len(xml.findall('./joint/mimic')) == 4 and len(xml.findall('transmission')) == 4
        assert abs(total-manifest['source_URDF_mass_kg']) < 1e-9
        loops = xml.findall('loop_joint')
        assert len(loops) == (4 if '_closed_isaac' in path.name else 0)
        trees.append(dict(file=path.name, links=len(links), tree_joints=len(joints), meshes=len(meshes), loop_extensions=len(loops)))
    assert len(trees) == 2
    triangles = 0
    for path in (robot/'meshes').glob('*.STL'):
        raw = path.read_bytes()
        count = struct.unpack_from('<I', raw, 80)[0]
        assert count > 0 and len(raw) == 84+count*50, path.name
        triangles += count
    scene_path = robot/'mujoco/dogleg_ground.xml'
    scene = ET.parse(scene_path).getroot()
    meshroot = scene_path.parent/scene.find('compiler').get('meshdir', '')
    scene_meshes = [relative_file(meshroot, node.get('file'), robot) for node in scene.findall('./asset/mesh')]
    assert len(scene_meshes) == 19
    payload = json.loads((project/'src/dogleg_rl/assets/physical_payload.json').read_text(encoding='utf-8'))
    assert payload['source_sha256'] == hashlib.sha256(scene_path.read_bytes()).hexdigest()
    assert payload['mass'] == manifest['training_runtime_mass_kg'] == 5.5
    assert len(payload['loops']) == len(payload['belts']) == 4
    return dict(passed=True, SHA_verified_files=len(entries), urdf=trees,
        source_STL_files=19, STL_triangles=triangles, source_scene_meshes=len(scene_meshes),
        training_mass_kg=5.5, source_URDF_mass_kg=manifest['source_URDF_mass_kg'],
        simulator_or_GPU_started=False)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
