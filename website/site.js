const pages = [...document.querySelectorAll('main > section')];
const tabs = [...document.querySelectorAll('.navigation a')];
const navigation = document.querySelector('.navigation');
navigation.setAttribute('role', 'tablist');
tabs.forEach((tab) => {
  tab.setAttribute('role', 'tab');
  tab.setAttribute('aria-controls', tab.hash.slice(1));
});
pages.forEach((page) => {
  page.setAttribute('role', 'tabpanel');
  page.setAttribute('aria-labelledby', `tab-${page.id}`);
  page.tabIndex = 0;
});
function showPage() {
  const id = location.hash.slice(1);
  const active = pages.some((page) => page.id === id) ? id : 'home';
  pages.forEach((page) => { page.hidden = page.id !== active; });
  tabs.forEach((tab) => {
    const selected = tab.hash === `#${active}`;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.title = `${tabs.find((tab) => tab.hash === `#${active}`).textContent} · LongMarch 长征`;
}
showPage();
window.addEventListener('hashchange', () => {
  if (!pages.some((page) => `#${page.id}` === location.hash)) return;
  showPage();
  window.scrollTo({ top: 0, behavior: 'instant' });
});
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
  tabs[next].click();
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
