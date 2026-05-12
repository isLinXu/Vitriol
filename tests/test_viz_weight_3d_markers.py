from pathlib import Path


def test_weight_3d_visualizer_caches_layer_list_dom_and_active_selection() -> None:
    html = Path("src/vitriol/viz/weight_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function byId(id)" in html
    assert "domCache" in html
    assert "layerListItemsByIndex" in html
    assert "activeLayerListItem" in html
    assert "document.createDocumentFragment()" in html
    assert "container.replaceChildren(fragment)" in html


def test_weight_3d_visualizer_disposes_instanced_mesh_resources_explicitly() -> None:
    html = Path("src/vitriol/viz/weight_3d_visualizer.html").read_text(encoding="utf-8")
    assert "function disposeInstancedMesh(mesh)" in html
    assert "mesh.geometry.dispose()" in html
    assert "mesh.material.dispose()" in html
    assert "disposeInstancedMesh(instancedMesh)" in html
