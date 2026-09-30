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
const tocLinks = [...document.querySelectorAll('.architecture-sidebar a')];
const architecture = document.querySelector('#architecture');
const headings = [...architecture.querySelectorAll('.anchor-heading')];
function highlightSection(id) {
  tocLinks.forEach((link) => {
    if (link.hash === `#${id}`) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
function showPage() {
  const target = document.getElementById(location.hash.slice(1));
  // The skip link moves focus to the current page without changing its tab.
  if (target?.id === 'content') return;
  const page = target?.closest('main > section') || pages[0];
  const deepLink = target && target !== page && page.contains(target);
  pages.forEach((item) => { item.hidden = item !== page; });
  tabs.forEach((tab) => {
    const selected = tab.hash === `#${page.id}`;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.title = `${tabs.find((tab) => tab.hash === `#${page.id}`).textContent} · LongMarch 长征`;
  if (deepLink) {
    const link = tocLinks.find((item) => item.hash === location.hash);
    const group = link?.closest('details');
    if (group) group.open = true;
    target.tabIndex = -1;
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: 'start' });
    highlightSection(target.id);
  } else {
    window.scrollTo({ top: 0, behavior: 'instant' });
    highlightSection('architecture');
    // Links inside a panel must not leave focus in a newly hidden panel.
    if (document.activeElement?.closest('section[hidden]')) page.focus({ preventScroll: true });
  }
}
showPage();
window.addEventListener('hashchange', showPage);
// Restore fragment positioning after the browser restores an old scroll offset.
window.addEventListener('pageshow', () => requestAnimationFrame(showPage));
let scrollScheduled = false;
window.addEventListener('scroll', () => {
  if (architecture.hidden || scrollScheduled) return;
  scrollScheduled = true;
  requestAnimationFrame(() => {
    scrollScheduled = false;
    let active = 'architecture';
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top > 100) break;
      active = heading.id;
    }
    highlightSection(active);
  });
}, { passive: true });
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
