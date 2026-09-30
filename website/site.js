const pages = [...document.querySelectorAll('main > section')];
const tabs = [...document.querySelectorAll('.navigation a')];
const navigation = document.querySelector('.navigation');
pages.forEach((page) => { page.tabIndex = -1; });
const legacyArchitecture = {
  architecture: 'index.html',
  'arch-grassland': 'modules/code/grassland/index.html',
  'arch-graphics': 'files/code/grassland/graphics/core.h.html',
  'arch-backends': 'modules/code/grassland/graphics/backend/index.html',
  'arch-shaders': 'files/code/grassland/graphics/shader.h.html',
  'arch-window': 'files/code/grassland/graphics/window.h.html',
  'arch-math': 'modules/code/grassland/math/index.html',
  'arch-bvh': 'modules/code/grassland/bvh/index.html',
  'arch-diff': 'modules/code/grassland/physics/diff_kernel/index.html',
  'arch-util': 'modules/code/grassland/util/index.html',
  'arch-sparkium': 'modules/code/sparkium/index.html',
  'arch-scene': 'files/code/sparkium/core/scene.h.html',
  'arch-pipelines': 'modules/code/sparkium/pipelines/index.html',
  'arch-geometry': 'modules/code/sparkium/geometry/index.html',
  'arch-materials': 'modules/code/sparkium/material/index.html',
  'arch-camera': 'modules/code/sparkium/camera/index.html',
  'arch-film': 'files/code/sparkium/core/film.h.html',
  'arch-scene-io': 'modules/code/sparkium/scene_io/index.html',
  'arch-snowberg': 'modules/code/snowberg/index.html',
  'arch-draw': 'modules/code/snowberg/draw/index.html',
  'arch-gui': 'modules/code/snowberg/gui/index.html',
  'arch-world-ui': 'files/code/snowberg/gui/world_panel.h.html',
  'arch-surface': 'modules/code/snowberg/gui/surface/index.html',
  'arch-visualizer': 'modules/code/snowberg/visualizer/index.html',
  'arch-solver': 'modules/code/snowberg/solver/index.html',
  'arch-simulation': 'modules/code/practium/index.html',
  'arch-pbd': 'files/code/contradium/pbd/pbd_solver.h.html',
  'arch-practium': 'modules/code/practium/index.html',
  'arch-python': 'modules/code/pybind/index.html',
};
function showPage() {
  const id = location.hash.slice(1);
  if (Object.hasOwn(legacyArchitecture, id)) {
    location.replace(`reference/${legacyArchitecture[id]}`);
    return;
  }
  if (id === 'content') return;
  const page = pages.find((item) => item.id === id) || pages[0];
  pages.forEach((item) => { item.hidden = item !== page; });
  tabs.forEach((tab) => {
    if (tab.hash === `#${page.id}`) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  });
  document.title = `${tabs.find((tab) => tab.hash === `#${page.id}`).textContent} · LongMarch 长征`;
  window.scrollTo({ top: 0, behavior: 'instant' });
  if (document.activeElement?.closest('section[hidden]')) page.focus({ preventScroll: true });
}
showPage();
window.addEventListener('hashchange', showPage);
window.addEventListener('pageshow', () => requestAnimationFrame(showPage));
navigation.addEventListener('keydown', (event) => {
  const index = tabs.indexOf(document.activeElement);
  if (index < 0) return;
  let next;
  if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
  if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
  if (event.key === 'Home') next = 0;
  if (event.key === 'End') next = tabs.length - 1;
  if (next === undefined) return;
  event.preventDefault();
  tabs[next].focus();
});
const common = 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release';
const commands = {
  mac: `${common} -DVCPKG_PATH=/path/to/vcpkg -DLONGMARCH_DISABLE_PYTHON=ON -DLONGMARCH_ENABLE_METAL=ON -DLONGMARCH_DISABLE_VULKAN=ON`,
  linux: `${common} -DVCPKG_PATH=/path/to/vcpkg -DLONGMARCH_DISABLE_PYTHON=ON`,
  windows: `${common} -DVCPKG_PATH=C:/path/to/vcpkg -DLONGMARCH_DISABLE_PYTHON=ON`,
};
document.querySelector('#platform').addEventListener('change', (event) => {
  document.querySelector('#configure').textContent = commands[event.target.value];
});
if (navigator.clipboard && window.isSecureContext) {
  document.querySelectorAll('pre').forEach((block) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy';
    button.textContent = '复制';
    button.setAttribute('aria-label', '复制此代码块');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(block.querySelector('code').textContent);
        button.textContent = '已复制';
        document.querySelector('#announcement').textContent = '命令已复制到剪贴板';
        setTimeout(() => { button.textContent = '复制'; }, 1800);
      } catch {
        button.textContent = '请手动复制';
      }
    });
    block.append(button);
  });
}
