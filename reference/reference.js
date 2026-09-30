// Documents are independent HTML pages; navigation never replaces a page with
// an in-page architecture jump. Filtering only changes the directory display.
const filter = document.querySelector('#filter');
const sidebar = document.querySelector('.doc-sidebar nav');
const links = [...sidebar.querySelectorAll('a')];
const groups = [...sidebar.querySelectorAll('details')];
let previousOpen = new Map();
filter.addEventListener('input', () => {
  const query = filter.value.trim().toLocaleLowerCase();
  if (query && previousOpen.size === 0) previousOpen = new Map(groups.map((group) => [group, group.open]));
  for (const link of links) link.hidden = Boolean(query && !link.textContent.toLocaleLowerCase().includes(query));
  for (const group of groups.toReversed()) {
    const titleMatch = group.querySelector('summary').textContent.toLocaleLowerCase().includes(query);
    if (query && titleMatch) group.querySelectorAll('a, details').forEach((item) => { item.hidden = false; });
    const hasMatch = [...group.querySelectorAll('a')].some((link) => !link.hidden);
    group.hidden = Boolean(query && !hasMatch && !titleMatch);
    if (query && !group.hidden) group.open = true;
    if (!query && previousOpen.has(group)) group.open = previousOpen.get(group);
  }
  if (!query) previousOpen.clear();
});
