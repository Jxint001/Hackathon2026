'use strict';
// Rendering and playback only. All plans, scores and block layouts are recorded.
const DEMOS = [CASE_05, CASE_08];
const $ = id => document.getElementById(id);
const fmt = (n, digits = 0) => Number(n).toLocaleString('en-US', {maximumFractionDigits: digits});
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors = ['#76a9ff', '#e5ae73'];
const modes = ['拓扑序', '反向 DFS', '关键路径优先'];
let caseIndex = 0, tab = -1, step = 0, timer = null, selectedNode = null;
let showCandidate = true, showAll = false;
let graphView = 'dag', selectedDagNode = null, dagZoom = .85;
let selectedGridCell = 0;

function model() { return DEMOS[caseIndex]; }
function usedCores(plan) { return plan.core_schedules.filter(seq=>seq.length>0).length; }
function opLabel(id) { return model().opNames[id]; }
function tensorLabel(id) { return model().tensorNames[id]; }
function nodeLabel(id) { return model().opNames[id] || model().tensorNames[id]; }
function opList(ids, separator='、') { return ids.map(opLabel).join(separator); }
const groupColor = g => `hsl(${(g*137.508+205)%360} 55% 70%)`;
function coreMap(plan) {
  const groups = {};
  plan.core_schedules.forEach((seq, c) => seq.forEach(g => { groups[g] = c; }));
  return Object.fromEntries(Object.entries(plan.node_to_subgraph).map(([op, g]) => [op, groups[g]]));
}
function groupOps(plan, order) {
  const groups = {};
  for (const op of order) {
    const g = plan.node_to_subgraph[op];
    if (g !== undefined) (groups[g] ??= []).push(op);
  }
  return groups;
}


function searches() { const m = model(); return showAll ? m.searches : m.focus.map(i => m.searches[i]); }
function maxStep() { return tab===-1?model().grouping.length-1:tab === 0 ? model().steps.length+1 : searches().length+1; }
function activeSearch() { return tab === 1 && step > 0 && step <= searches().length ? searches()[step-1] : null; }
function stop() { if (timer) clearInterval(timer); timer = null; $('play').textContent = '▶ 自动演示'; }
function setStep(value) {
  const previous=step;step = Math.max(0,Math.min(maxStep(),value));
  if(tab===0&&previous===0&&step>0)graphView='grouped';
  if(tab===-1){graphView=step===0?'dag':'grouped';selectedNode=null;selectedDagNode=null;}
  render();
}
function setTab(value) { stop(); tab = value; step = 0; selectedNode = null; selectedDagNode = null; showCandidate = true; graphView=value<=0?'dag':'grouped'; render(); }
function setCase(value) { stop(); selectedGridCell=0;caseIndex=value; step=0; selectedNode=null; selectedDagNode=null; showCandidate=true; graphView=tab<=0?'dag':'grouped';render(); }

function displayState() {
  const m = model();
  if(tab===-1){
    const e=m.grouping[step],mapping={};
    e.groups.forEach((ops,g)=>ops.forEach(op=>{mapping[op]=g;}));
    return {plan:{node_to_subgraph:mapping,core_schedules:m.initial.plan.core_schedules.map(()=>[])},order:m.compute_ids,focus:e.focus||[],changed:[]};
  }
  if (tab === 0) {
    const count = Math.min(step, m.steps.length);
    const scheduled = new Set(m.steps.slice(0,count).map(s => s.group));
    const plan = {node_to_subgraph:m.initial.plan.node_to_subgraph,
      core_schedules:m.initial.plan.core_schedules.map(seq => seq.filter(g => scheduled.has(g)))};
    return {plan, order:m.initial.order.flatMap(g => m.initial.groups[g].ops),
      focus: step > 0 && step <= m.steps.length ? m.initial.groups[m.steps[step-1].group].ops : [], changed:[]};
  }
  const e = activeSearch();
  if (!e) return {plan:step === 0 ? m.initial.plan:m.final.plan,
    order:step === 0 ? m.initial.order.flatMap(g => m.initial.groups[g].ops):m.finalOrder, focus:selectedGridOps(), changed:[]};
  const before = coreMap(e.beforePlan), after = coreMap(e.plan);
  const changed = m.compute_ids.filter(op => before[op] !== after[op]);
  const moved=e.recipe?.kind==='lifetime'?e.recipe.moves.map(x=>x.op):e.recipe?.kind==='chain'?e.recipe.changes.map(x=>x.op):[];
  return {plan:showCandidate ? e.plan:e.beforePlan, order:showCandidate ? e.order:e.beforeOrder, focus:[...new Set([...moved,...selectedGridOps()])], changed:[...new Set([...changed,...moved])]};
}

function gridExplainerVisible() {
  return caseIndex===0&&tab===1&&!!model().grid&&(step===0||activeSearch()?.stage==='grid_search');
}
function selectedGridOps() {
  return gridExplainerVisible()?model().grid.cells[selectedGridCell]||model().grid.cells[0]:[];
}
// Read the recorded block layout for this candidate.
function gridTiles(m,e) { return e.gridLayout; }
function selectGridCell(index) {
  selectedGridCell=index;selectedNode=null;selectedDagNode=null;graphView='grouped';render();
}
function jumpGridCandidate(multi) {
  const m=model(),candidates=m.searches.filter(e=>e.stage==='grid_search');
  const matching=candidates.filter(e=>multi?gridTiles(m,e).tiles.some(t=>t.cells.length>1):e.recipe.rows===1&&e.recipe.cols===1&&e.valid);
  const e=matching.find(e=>e.accepted)||matching[0];
  if(!e)return;
  stop();if(!searches().includes(e))showAll=true;
  showCandidate=true;setStep(searches().indexOf(e)+1);
}
function renderGridExplainer() {
  const panel=$('grid-explainer');panel.hidden=!gridExplainerVisible();
  if(panel.hidden)return;
  const m=model(),grid=m.grid,e=activeSearch();
  selectedGridCell=Math.min(selectedGridCell,grid.cells.length-1);
  const row=Math.floor(selectedGridCell/grid.cols),col=selectedGridCell%grid.cols;
  const active=e&&showCandidate,layout=active?gridTiles(m,e):null;
  const tileFor=new Map();if(layout)for(const tile of layout.tiles)for(const cell of tile.cells)tileFor.set(cell,tile);
  const assignment=coreMap(displayState().plan),chain=grid.cells[selectedGridCell];
  const tile=tileFor.get(selectedGridCell);
  $('grid-layout-note').innerHTML=active?
    `<b>本次候选：目标分块 ${e.recipe.rows}×${e.recipe.cols}，使用 ${e.recipe.active} 核</b><br>虚线围成一个计算块，共 ${layout.tiles.length} 块；块内包含的格子一起组织执行。填色表示核心，亮实线框标出选中的格子。`:
    `<b>${grid.cells.length} 条乘加链 → ${grid.rows} 行 × ${grid.cols} 列的输出单元网格</b><br>${e?'当前显示比较前方案，尚未套用本次候选分块。':'先看对应关系，再用上方按钮查看真实分块候选。'}点击任意格子，查看下面对应的 DAG 操作链。`;
  const left=115,top=90,w=164,h=110,width=left+grid.cols*w+14,height=top+grid.rows*h+10;
  let svg=`<text x="${left}" y="20" fill="#9aafc2" font-size="11">行、列标签：各链首个乘法共享的输入张量</text>`;
  for(let c=0;c<grid.cols;c++) {
    const x=left+c*w+w/2,color=c===col?'#b6ef73':'#9aafc2';
    svg+=`<text x="${x}" y="48" text-anchor="middle" fill="${color}" font-size="12">第 ${c+1} 列 · 共享</text><text x="${x}" y="68" text-anchor="middle" fill="${color}" font-size="12" font-family="Consolas,monospace">${tensorLabel(grid.colInputs[c])}</text>`;
  }
  for(let r=0;r<grid.rows;r++) {
    const y=top+r*h+h/2,color=r===row?'#b6ef73':'#9aafc2';
    svg+=`<text x="12" y="${y-8}" fill="${color}" font-size="12">第 ${r+1} 行 · 共享</text><text x="12" y="${y+13}" fill="${color}" font-size="12" font-family="Consolas,monospace">${tensorLabel(grid.rowInputs[r])}</text>`;
  }
  grid.cells.forEach((ops,i)=>{
    const r=Math.floor(i/grid.cols),c=i%grid.cols,x=left+c*w,y=top+r*h,selected=i===selectedGridCell;
    const core=assignment[ops[0]],color=colors[core],t=tileFor.get(i);
    svg+=`<g data-grid-cell="${i}" role="button" tabindex="0" aria-pressed="${selected}" aria-label="第 ${r+1} 行第 ${c+1} 列，${opList(ops)}，核 ${core}"><rect x="${x+8}" y="${y+8}" width="${w-16}" height="${h-16}" rx="7" fill="${color}20" stroke="${selected?'#b6ef73':color+'90'}" stroke-width="${selected?3:1}"/>
      <text x="${x+w/2}" y="${y+30}" text-anchor="middle" fill="#d3dfed" font-size="12">输出单元 (${r+1}, ${c+1})</text>
      <text x="${x+w/2}" y="${y+52}" text-anchor="middle" fill="${color}" font-size="10.5" font-family="Consolas,monospace">${ops.map(opLabel).join(' → ')}</text>
      <text x="${x+w/2}" y="${y+76}" text-anchor="middle" fill="${color}" font-size="12">核 ${core}${t?' · 块 '+(t.index+1):' · 一条乘加链'}</text>
      <title>点击查看 ${opList(ops)} 在 DAG 中的依赖</title></g>`;
  });
  if(layout)for(const t of layout.tiles) {
    const members=new Set(t.cells);let path='';
    for(const cell of t.cells){
      const r=Math.floor(cell/grid.cols),c=cell%grid.cols,x=left+c*w,y=top+r*h;
      if(r===0||!members.has(cell-grid.cols))path+=`M${x},${y}h${w}`;
      if(r===grid.rows-1||!members.has(cell+grid.cols))path+=`M${x},${y+h}h${w}`;
      if(c===0||!members.has(cell-1))path+=`M${x},${y}v${h}`;
      if(c===grid.cols-1||!members.has(cell+1))path+=`M${x+w},${y}v${h}`;
    }
    svg+=`<path data-grid-tile="${t.index}" d="${path}" fill="none" stroke="${t===tile?'#b6ef73':'#adbacb'}" stroke-width="2" stroke-dasharray="6 4" pointer-events="none"/>`;
  }
  $('grid-map').setAttribute('viewBox',`0 0 ${width} ${height}`);$('grid-map').innerHTML=svg;
  $('grid-map').querySelectorAll('[data-grid-cell]').forEach(el=>{
    const select=()=>selectGridCell(Number(el.dataset.gridCell??el.dataset['grid-cell']));
    el.addEventListener('click',select);el.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();select();}});
  });
  const output=m.outgoing[chain.at(-1)].find(t=>m.tensors[t]);
  let chainHTML='';
  chain.forEach((op,i)=>{
    chainHTML+=`<button class="grid-chain-op" data-grid-op="${op}"><b>${opLabel(op)}</b><small>${esc(m.ops[op].op)}</small></button>`;
    if(i<chain.length-1){const t=m.outgoing[op].find(t=>(m.incoming[chain[i+1]]||[]).includes(t));chainHTML+=`<span class="grid-chain-link"><small>${tensorLabel(t)}</small><b>→</b></span>`;}
  });
  $('grid-chain').innerHTML=`<div class="grid-chain-heading"><b>选中一格，展开它在 DAG 中的链</b><span>输出单元 (${row+1}, ${col+1})</span></div><p class="micro">首个乘法读取 ${tensorLabel(grid.rowInputs[row])} 和 ${tensorLabel(grid.colInputs[col])}；后续乘加还会读取各自的输入张量。</p><div class="grid-chain-flow">${chainHTML}<span class="grid-chain-output">→ 结果张量<br><b>${tensorLabel(output)}</b></span></div>`;
  $('grid-chain').querySelectorAll('[data-grid-op]').forEach(el=>el.addEventListener('click',()=>{selectedNode=Number(el.dataset.gridOp??el.dataset['grid-op']);graphView='grouped';renderGraph();}));
  if(tile){
    const firstLayer=tile.cells.map(i=>grid.cells[i][0]);
    $('grid-tile-order').innerHTML=`<b>选中格子所在的块 ${tile.index+1}：${tile.cells.length} 个输出单元，${tile.ops.length} 个操作，放到核 ${tile.core}</b>
      <p>${tile.cells.length===1?'1×1 分块：这个框只包含一格，也就是一条完整乘加链。':`本块把 ${tile.cells.length} 条链放在一起。先做 ${opList(firstLayer,'、')}，再推进各链的下一层。`}</p>
      <div class="grid-block-order">${tile.ops.map((u,i)=>`${i?'<span>→</span>':''}<code>${opLabel(u)}</code>`).join('')}</div>
      <p class="micro">“计算块”用于组织候选操作顺序；PLACEMENT 中的框是提交模拟的子图。本次候选按单操作子图提交，计算块与提交子图不是同一概念。</p>`;
  }else $('grid-tile-order').innerHTML='<p>同一行的链共享行标签输入，同一列的链共享列标签输入，所以代码能从 DAG 识别出行列结构。点上方“查看多单元分块”，观察几条链如何被组织进同一个计算块。</p>';
}

function dagData(m, expanded) {
  const compute = new Set(m.compute_ids);
  const ids = expanded ? new Set([...m.graph.ops,...m.graph.tensors].map(n=>n.id))
    : new Set([...compute,...m.compute_ids.flatMap(u=>[...(m.incoming[u]||[]),...(m.outgoing[u]||[])])]);
  const edges=m.graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));
  const incoming=new Map([...ids].map(id=>[id,[]])),outgoing=new Map([...ids].map(id=>[id,[]]));
  for(const e of edges){incoming.get(e.target).push(e.source);outgoing.get(e.source).push(e.target);}
  const degree=new Map([...ids].map(id=>[id,incoming.get(id).length]));
  const rank=new Map([...ids].map(id=>[id,0]));
  const queue=[...ids].filter(id=>degree.get(id)===0).sort((a,b)=>a-b);
  for(let i=0;i<queue.length;i++)for(const v of outgoing.get(queue[i])){
    rank.set(v,Math.max(rank.get(v),rank.get(queue[i])+1));
    degree.set(v,degree.get(v)-1);if(degree.get(v)===0)queue.push(v);
  }
  if(queue.length!==ids.size)throw new Error('DAG contains a cycle');
  // Place external inputs beside the first operation that needs them, so later
  // reduction inputs do not draw long edges across earlier computation columns.
  for(const id of [...queue].reverse())if(!compute.has(id)&&outgoing.get(id).length){
    rank.set(id,Math.min(...outgoing.get(id).map(v=>rank.get(v)))-1);
  }
  const layers=[];
  for(const id of queue)(layers[rank.get(id)]??=[]).push(id);
  const order=new Map();
  const refresh=()=>layers.forEach(layer=>layer.forEach((id,i)=>order.set(id,i)));
  refresh();
  // Alternating barycenter sweeps keep shared inputs and their readers close.
  for(let pass=0;pass<6;pass++){
    const forward=pass%2===0,indices=layers.map((_,i)=>i);
    if(!forward)indices.reverse();
    for(const i of indices){
      const score=id=>{const neighbors=(forward?incoming:outgoing).get(id);return neighbors.length?neighbors.reduce((sum,n)=>sum+order.get(n),0)/neighbors.length:order.get(id);};
      layers[i].sort((a,b)=>score(a)-score(b)||order.get(a)-order.get(b)||a-b);refresh();
    }
  }
  const width=layers.length*190+35,height=Math.max(...layers.map(l=>l.length))*65+75;
  const positions=new Map();
  layers.forEach((layer,col)=>layer.forEach((id,row)=>positions.set(id,{x:24+col*190,y:45+row*65+(height-75-layer.length*65)/2})));
  return {ids,edges,incoming,outgoing,layers,positions,width,height};
}

function renderDag() {
  const m=model(),expanded=$('expand-boundaries').checked,d=dagData(m,expanded),svg=$('graph');
  if(selectedDagNode!==null&&!d.ids.has(selectedDagNode))selectedDagNode=null;
  svg.setAttribute('viewBox',`0 0 ${d.width} ${d.height}`);
  svg.setAttribute('style',`width:${Math.round(d.width*dagZoom)}px;height:${Math.round(d.height*dagZoom)}px;max-height:none;min-height:0`);
  svg.setAttribute('aria-label',`${m.id} 简化用例子图划分前的操作—张量 DAG`);
  const related=new Set(selectedDagNode===null?[]:[selectedDagNode,...d.incoming.get(selectedDagNode),...d.outgoing.get(selectedDagNode)]);
  const selectedIsTensor=selectedDagNode!==null&&!!m.tensors[selectedDagNode];
  // For a selected operation, also show the other readers/writers of its tensors.
  if(selectedDagNode!==null&&!selectedIsTensor)for(const t of [...d.incoming.get(selectedDagNode),...d.outgoing.get(selectedDagNode)]){
    for(const u of [...d.incoming.get(t),...d.outgoing.get(t)])related.add(u);
  }
  let html='<defs><marker id="dag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10Z" fill="#5d738c"/></marker><marker id="dag-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10Z" fill="#b6ef73"/></marker></defs>';
  d.layers.forEach((layer,i)=>{html+=`<text x="${24+i*190}" y="22" fill="#778ba4" font-size="11">拓扑层 ${i}</text>`;});
  for(const {source,target} of d.edges){
    const a=d.positions.get(source),b=d.positions.get(target),focus=related.has(source)&&related.has(target);
    const opacity=selectedDagNode===null||focus?1:.15;
    html+=`<path data-dag-edge="${source}:${target}" class="edge${focus?' focus':''}" opacity="${opacity}" d="M${a.x+142},${a.y+22} C${a.x+168},${a.y+22} ${b.x-26},${b.y+22} ${b.x-2},${b.y+22}" marker-end="url(#dag-${focus?'active':'arrow'})"><title>${nodeLabel(source)} → ${nodeLabel(target)}</title></path>`;
  }
  for(const id of d.ids){
    const p=d.positions.get(id),tensor=m.tensors[id],op=m.ops[id],copy=op&&['COPY_IN','COPY_OUT'].includes(op.op);
    const boundary=expanded?false:(d.incoming.get(id).length===0||d.outgoing.get(id).length===0);
    const readers=tensor?d.outgoing.get(id).length:0,writers=tensor?d.incoming.get(id).length:0;
    const color=tensor?(readers>1||writers>1?'#e5ae73':'#70c9c0'):copy?'#7f8b9e':'#76a9ff';
    const selected=id===selectedDagNode,opacity=selectedDagNode===null||related.has(id)?1:.28;
    const label=nodeLabel(id);
    const subtitle=tensor?`${tensor.pos} · ${fmt(tensor.size/1024,2)} KiB`:op.op;
    html+=`<g class="op-node dag-node" data-dag-node="${id}" tabindex="0" role="button" aria-label="${esc(label+' '+subtitle)}" opacity="${opacity}" transform="translate(${p.x} ${p.y})"><rect width="142" height="44" rx="${tensor?20:5}" fill="${color}17" stroke="${selected?'#b6ef73':color}" stroke-width="${selected?2.5:1}"${copy?' stroke-dasharray="4 3"':''}/><text x="12" y="18" fill="${color}" font-size="12" font-family="Consolas,monospace">${label}</text><text x="12" y="34" fill="#b4c2d3" font-size="10" font-family="Consolas,monospace">${esc(subtitle)}</text>`;
    if(tensor&&(readers>1||writers>1))html+=`<text x="134" y="-5" text-anchor="end" fill="#e5ae73" font-size="10">${writers>1?writers+' 写者':readers+' 读取者'}</text>`;
    if(boundary&&tensor)html+=`<text x="10" y="-5" fill="#8ba5b1" font-size="9">${d.incoming.get(id).length?'输出边界':'输入边界'}</text>`;
    html+=`<title>${esc(label+' '+subtitle)}</title></g>`;
  }
  svg.innerHTML=html;
  svg.querySelectorAll('[data-dag-node]').forEach(el=>{
    const select=()=>{const id=Number(el.dataset['dag-node']??el.dataset.dagNode);selectedDagNode=selectedDagNode===id?null:id;renderGraph();};
    el.addEventListener('click',select);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
  });
  $('graph-title').textContent='简化用例的初始 DAG · 子图划分前';
  $('graph-caption').textContent=`${d.ids.size} 个节点 · ${d.edges.length} 条边 · 箭头表示“写入张量”或“读取张量” · 可横向 / 纵向滚动`;
  $('dag-guide').innerHTML=`${expanded?'已展开小图 JSON 中的全部节点与边，包括缩图时补入的 COPY / DDR 边界。':'已展开所有计算操作和其读写张量；COPY / DDR 边界暂时折叠。勾选上方选项可查看完整小图 JSON。'}<br>${caseIndex===0?`试着点击 <button class="text-button" id="dag-example">${tensorLabel(39364)}</button>：它被三个 MUL 读取，属于共享输入，不会产生三个 MUL 之间的先后依赖。`:`试着点击 <button class="text-button" id="dag-example">${tensorLabel(41205)}</button>：可看到双写者与两个卷积读取者；写者需要绑定，两个读取者不因此互相依赖。`}`;
  $('dag-example').addEventListener('click',()=>{selectedDagNode=caseIndex===0?39364:41205;renderGraph();});
  if(selectedDagNode===null)$('node-detail').textContent='圆角胶囊是张量，矩形是操作；橙色张量有多个读取者或写者。点击节点高亮其关联关系，再次点击取消。当前展示的是两个简化用例在子图划分前的 DAG。';
  else{
    const id=selectedDagNode,t=m.tensors[id],o=m.ops[id];
    const label=nodeLabel;
    $('node-detail').innerHTML=`<strong>${t?'张量 '+tensorLabel(id):opLabel(id)+' · '+esc(o.op)}</strong>　${t?esc(t.pos)+' · '+fmt(t.size)+' bytes':esc(o.pipe)+' · '+(o.cycles<0?'按搬运量计算周期':fmt(o.cycles)+' cycles')}<div class="node-tensors">${t?'写者':'输入张量'}：${d.incoming.get(id).map(label).join('、')||'输入边界'}<br>${t?'读取者':'输出张量'}：${d.outgoing.get(id).map(label).join('、')||'输出边界'}${t&&d.incoming.get(id).length>1?'<br>多个计算写者必须同核；这是共享写者绑定的依据。':''}${t&&d.outgoing.get(id).length>1?'<br>多个读取者共享这份数据，但不因此互相依赖。':''}</div>`;
  }
}

function renderGraph() {
  const raw=graphView==='dag';
  $('view-dag').classList.toggle('active',raw);$('view-grouped').classList.toggle('active',!raw);
  $('view-dag').setAttribute('aria-pressed',raw);$('view-grouped').setAttribute('aria-pressed',!raw);
  $('dag-options').hidden=!raw;$('group-option').hidden=raw;$('dag-guide').hidden=!raw;
  $('graph-wrap').classList.toggle('raw-dag',raw);
  $('graph-legend').innerHTML=raw?'<span><i class="c0"></i>操作</span><span><i style="background:#70c9c0"></i>张量</span><span><i class="c1"></i>共享张量</span>':'<span><i class="c0"></i>核 0</span><span><i class="c1"></i>核 1</span>';
  if(!raw&&tab===-1)$('graph-legend').innerHTML='<span>虚线框：已合并的子图</span><span>亮边框：本次合并</span>';
  if(raw){renderDag();return;}
  $('graph').setAttribute('style','');$('graph').setAttribute('aria-label','计算操作依赖与子图划分图');
  const m = model(), state = displayState(), assignment = coreMap(state.plan);
  const pos = {}, width = 770, nodeW = 126, nodeH=39;
  const rowGap=55, firstY=38;
  if (caseIndex === 0) {
    m.lanes.forEach((lane,r) => lane.forEach((op,c) => {pos[op]={x:100+c*231,y:firstY+r*rowGap};}));
  } else {
    m.lanes.slice(0,-1).forEach((lane,r) => lane.forEach((op,c) => {pos[op]={x:22+c*185,y:firstY+r*rowGap};}));
    m.lanes[m.lanes.length-1].forEach((op,i) => {pos[op]={x:620,y:150+i*195};});
  }
  const height=caseIndex === 0 ? 555:500;
  const svg=$('graph'); svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  let html='<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#5e728b"/></marker><marker id="arrow-focus" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b6ef73"/></marker></defs>';
  const headings=caseIndex === 0 ? [[100,'第一步 · A × B'],[331,'第二步 · 累加'],[562,'第三步 · 累加']] : [[22,'ADDS'],[207,'CAST'],[392,'UB_COPY_L1'],[620,'CONV']];
  for(const [x,label] of headings) html+=`<text x="${x}" y="17" fill="#7f91a7" font-size="10" font-family="Consolas, monospace">${label}</text>`;
  if(tab===-1)model().grouping[step].groups.forEach((members,g)=>{
    if(members.length<2)return;
    const points=members.map(id=>pos[id]),x=Math.min(...points.map(p=>p.x))-7,y=Math.min(...points.map(p=>p.y))-7;
    const w=Math.max(...points.map(p=>p.x))+nodeW+7-x,h=Math.max(...points.map(p=>p.y))+nodeH+7-y;
    html+=`<rect data-merge-group="${g}" x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="none" stroke="${groupColor(g)}" stroke-dasharray="4 4" opacity=".65"/>`;
  });
  for(const {u,v,tensor} of m.edges) {
    const a=pos[u], b=pos[v], focus=selectedNode===u||selectedNode===v;
    const x1=a.x+nodeW+2,y1=a.y+nodeH/2,x2=b.x-3,y2=b.y+nodeH/2;
    html+=`<path class="edge${focus?' focus':''}" d="M${x1},${y1} C${x1+35},${y1} ${x2-35},${y2} ${x2},${y2}" marker-end="url(#arrow${focus?'-focus':''})"><title>${tensorLabel(tensor)} · ${fmt(m.tensors[tensor].size)} bytes</title></path>`;
  }
  const showGroups=$('show-groups').checked;
  for(const id of m.compute_ids) {
    const o=m.ops[id],p=pos[id],c=assignment[id],g=state.plan.node_to_subgraph[id];
    const color=tab===-1?groupColor(g):c===undefined?'#53647a':colors[c];
    const isFocus=state.focus.includes(id),selected=selectedNode===id,changed=state.changed.includes(id);
    const name=o.op==='A_MULACC_B'?'MULACC':o.op==='A_MUL_B'?'MUL':o.op;
    html+=`<g class="op-node" data-node="${id}" tabindex="0" role="button" aria-label="操作 ${opLabel(id)}，${esc(o.op)}，${c===undefined?'尚未分核':'核心 '+c}" transform="translate(${p.x} ${p.y})">`;
    if(isFocus||selected) html+=`<rect x="-4" y="-4" width="${nodeW+8}" height="${nodeH+8}" rx="8" fill="none" stroke="${selected?'#b6ef73':color}" opacity=".65"/>`;
    html+=`<rect width="${nodeW}" height="${nodeH}" rx="5" fill="${c===undefined?'#18212d':color+'1b'}" stroke="${color}" stroke-width="${isFocus?2:1}"/>`;
    html+=`<text x="9" y="15" fill="${c===undefined?'#d0d9e5':color}" font-size="11" font-family="Consolas,monospace">${opLabel(id)}</text>`;
    if(showGroups) html+=`<text x="${nodeW-8}" y="14" text-anchor="end" fill="#8f9db0" font-size="9" font-family="Consolas,monospace">G${g}</text>`;
    html+=`<text x="9" y="30" fill="#9aaabd" font-size="9" font-family="Consolas,monospace">${name} · ${o.cycles}</text>`;
    if(changed) html+=`<circle cx="${nodeW-8}" cy="27" r="3" fill="#b6ef73"/>`;
    html+='<title>'+esc(`${opLabel(id)} · ${o.op} · ${o.cycles} cycles` )+'</title></g>';
  }
  svg.innerHTML=html;
  $('graph-title').textContent=tab===-1?`从计算依赖到子图划分 · 当前 ${m.grouping[step].groups.length} 个子图`:tab===0?'计算图与初始子图':activeSearch()?(showCandidate?'搜索候选的分核与子图划分':'搜索前保留的方案'):'计算图与分核结果';
  $('graph-caption').textContent=tab===-1?'只保留计算依赖；合并后的操作仍展开显示，虚线框对应下方子图划分成员。':tab===0?'实线：通过张量传递的计算依赖 · G：初始子图':'绿色圆点：本次移动的操作 · G：当前子图';
  svg.querySelectorAll('[data-node]').forEach(el=>{
    const select=()=>{selectedNode=Number(el.dataset.node);renderGraph();};
    el.addEventListener('click',select);
    el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
  });
  if(selectedNode!==null) {
    const id=selectedNode,o=m.ops[id],c=assignment[id];
    const ins=(m.incoming[id]||[]).map(t=>{
      const readers=(m.outgoing[t]||[]).filter(u=>m.compute_ids.includes(u));
      const writers=(m.incoming[t]||[]).filter(u=>m.compute_ids.includes(u));
      return `${tensorLabel(t)} (${fmt(m.tensors[t].size)} B${readers.length>1?'，读取者 '+opList(readers,'/') :''}${writers.length>1?'，写者 '+opList(writers,'/') :''})`;
    });
    $('node-detail').innerHTML=`<strong>${opLabel(id)} · ${esc(o.op)}</strong>　${fmt(o.cycles)} cycles · ${esc(o.pipe)} · ${c===undefined?'尚未分核':'核 '+c}<div class="node-tensors">输入：${ins.join('；')}<br>前驱：${m.edges.filter(e=>e.v===id).map(e=>opLabel(e.u)).join('、')||'边界输入'}。点击其他操作可继续查看。</div>`;
  } else $('node-detail').textContent='点击操作，查看操作名称、计算周期、依赖和共享张量。COPY 边界已折叠，图中展示全部计算操作。';
}

function renderQueues() {
  if(tab===-1){
    const e=model().grouping[step];
    $('queue-title').textContent=`当前划分：${e.groups.length} 个子图`;
    $('queues').innerHTML=`<div class="grouping-members">${e.groups.map((members,g)=>`<div class="sg-block" style="border-color:${groupColor(g)}">${members.map(op=>`<span class="op-chip${(e.focus||[]).includes(op)?' changed':''}">${opLabel(op)}</span>`).join('')}</div>`).join('')}</div>`;
    $('queue-note').textContent='每个框表示一个计算子图，每个操作恰好属于一个子图。这里确定哪些操作合并，还没有分配核心；最终划分与 C++ 初始子图一致。';
    return;
  }
  const state=displayState(), groups=groupOps(state.plan,state.order);
  $('queues').innerHTML=state.plan.core_schedules.map((seq,c)=>`<div class="core-row"><span class="core-label" style="color:${colors[c]}">核 ${c}</span><div class="core-seq">${seq.length?seq.map(g=>`<span class="sg-block" style="border-color:${colors[c]}55">${(groups[g]||[]).map(op=>`<span class="op-chip${state.changed.includes(op)?' changed':''}" title="${esc(model().ops[op].op)}">${opLabel(op)}</span>`).join('')}</span>`).join(''):'<span class="empty-core">'+(tab===0&&step<maxStep()?'尚未分配':'空闲')+'</span>'}</div></div>`).join('');
  $('queue-title').textContent=tab===0?'逐步构造核心队列':'候选子图与操作顺序';
  $('queue-note').textContent='每个框是一张子图，框内为操作编号；这是方案顺序，不是执行时间线。同核子图会合并成 Task 后再执行核内调度。';
}

function sourceRef(path,func) { const [file,line]=path.split(':'); return `<div class="source-ref"><span>${esc(func)}</span><a href="../npu_scheduler/src/${file}" target="_blank" rel="noopener">${esc(file)} ↗</a></div>`; }
function stats(before,after) { return `<div class="stats-pair"><div><small>当前保留方案 / cycles</small><strong>${fmt(before)}</strong></div><div class="${after<before?'good':''}"><small>候选完整模拟 / cycles</small><strong>${fmt(after)}</strong></div></div>`; }
const groupingTitles={dag:'子图划分前：先看完整的数据依赖',projection:'把张量连接转换为计算依赖',writers:'步骤 1：检查共享写者，按需合并',no_writers:'步骤 1：检查共享写者，按需合并',cycle:'步骤 2：检查子图间环，按需合并',no_cycles:'步骤 2：检查子图间环，按需合并',chain_rule:'步骤 3：检查短链条件，满足则合并',chain:'步骤 3：执行一次短链合并',result:'得到初始子图'};
function groupingDetail(){
  const m=model(),e=m.grouping[step],first=m.edges[0];
  const mergeCount=m.grouping.filter(x=>x.kind==='chain').length,writers=m.grouping.filter(x=>x.kind==='writers').length;
  let content='';
  if(e.kind==='dag')content=`<p class="detail-description">${m.compute_ids.length} 个计算操作最初各自构成一个子图。操作以 <strong>op_1、op_2…</strong> 连续命名，张量以 <strong>tensor_1、tensor_2…</strong> 连续命名，图、表格与搜索过程始终使用相同名称。</p><div class="callout">${opLabel(first.u)} → ${tensorLabel(first.tensor)} → ${opLabel(first.v)}<br>表示前一个操作写出张量，后一个操作读取它。后者必须等待前者完成。</div><p class="detail-description">先分清两件事：<strong>共享读取</strong>是一份输入被多处使用；<strong>共享写入</strong>是多个操作共同写一份张量。代码只强制绑定共享写者。</p><p class="micro">COPY 节点负责边界搬运，不参与初始计算子图划分。点击“下一步”，观察如何从张量 DAG 推出计算依赖。</p>`;
  else if(e.kind==='projection')content=`<p class="detail-description">对每个张量，将它的计算写者连接到每个计算读取者。折叠张量节点和边界 COPY 后，得到左侧的操作依赖图。</p><div class="formula">${opLabel(first.u)} → ${tensorLabel(first.tensor)} → ${opLabel(first.v)}\n                    ↓\n              ${opLabel(first.u)} → ${opLabel(first.v)}</div><div class="callout">共享外部输入不产生读取者之间的边。${caseIndex===0?'MULACC 可以有三个输入张量，但只有一个计算前驱：另两个输入来自图边界。':'两个 UB_COPY_L1 共同写入卷积输入张量，它们都连到后续 CONV；两个写者之间不因此产生先后依赖。'}</div><p class="micro">接下来依次执行共享写者绑定、子图间环检查、短链合并。当前仍为 ${e.groups.length} 个单操作子图。</p>`;
  else if(e.kind==='writers')content=`<div class="callout">约束【写同一张量的计算操作必须同核】</div><p class="detail-description">${opList(e.focus)} 共同写入 <strong>${tensorLabel(e.tensor)}</strong>。为满足同核约束，将它们所在的子图合并。</p><div class="callout success">本次合并成员：${opList(e.focus)}<br>子图数量：${e.before.length} → ${e.groups.length}</div><p class="micro">共享读取不触发这一约束。</p>`;
  else if(e.kind==='no_writers')content='<div class="callout">约束【写同一张量的计算操作必须同核】</div><p class="detail-description">本例没有多个计算操作共同写入同一张量，因此不发生合并。共享读取不触发这一约束。</p>';
  else if(e.kind==='no_cycles')content=`<div class="callout">约束【禁止成环】——子图间依赖必须无环。</div><p class="detail-description">沿用<strong>步骤 1 处理后的子图划分</strong>，检查更新后的子图间依赖，不重新从单 op 子图开始。每个 op 单独构成子图时原本无环，但合并共享写者后，收缩得到的子图依赖图可能出现环。</p><p class="detail-description">本例检查后没有子图间环，不发生合并，仍为 <strong>${e.groups.length} 个子图</strong>。</p>`;
  else if(e.kind==='cycle')content=`<div class="callout">约束【禁止成环】——子图间依赖必须无环。</div><p class="detail-description">检查步骤 1 处理后的子图划分，发现子图间环，继续合并环上的子图。</p><div class="callout success">本次成员：${opList(e.focus)}。子图数量 ${e.before.length} → ${e.groups.length}。</div>`;
  else if(e.kind==='chain_rule')content=`<p class="detail-description">前两步保证约束满足后，采用<strong>短链合并</strong>简化初始化。沿一条依赖 u → v，只有同时满足以下四个条件，才合并它们所在的子图：</p><div class="formula">u 的后继子图数量 = 1\nv 的前驱子图数量 = 1\nu、v 当前属于不同子图\n两个子图的操作总数 ≤ 64</div><div class="callout">前两个条件表示连接处没有分叉或汇合；第三个条件避免重复合并；第四个条件限制子图的大小。这里统计的是<strong>计算依赖的子图间连接</strong>，不是输入张量数量。</div><p class="micro">这是简化初始化的启发式策略，64 是算法设定的阈值，不是题目约束。接下来逐次回放本例的 ${mergeCount} 次短链合并。</p>`;
  else if(e.kind==='chain')content=`<p class="detail-description">检查依赖 <strong>${opLabel(e.edge[0])} → ${opLabel(e.edge[1])}</strong>。在本轮扫描的邻接表中，源节点只有 ${e.successor_count} 个后继子图，目标节点只有 ${e.predecessor_count} 个前驱子图。</p><div class="formula">左子图：{ ${opList(e.left)} }\n右子图：{ ${opList(e.right)} }\n合并后包含 ${e.focus.length} 个操作，未超过上限 64\n→ { ${opList(e.focus)} }</div><div class="callout success">满足所有条件，合并一次。子图数量 ${e.before.length} → ${e.groups.length}。</div><p class="micro">亮边框标出本次涉及的操作，虚线框标出合并后的成员。合并会保留子图内计算依赖，不会把子图内操作变成同时执行。</p>`;
  else content=`<p class="detail-description">从 ${m.compute_ids.length} 个单操作子图开始，共享写者阶段合并 ${writers} 次，短链阶段合并 ${mergeCount} 次，得到 <strong>${m.initial.groups.length} 个初始子图</strong>。</p><div class="formula">${m.compute_ids.length} − ${writers} − ${mergeCount} = ${m.initial.groups.length} 个子图</div><div class="callout success">${caseIndex===0?'每条 MUL → MULACC → MULACC 链需要两次合并。九条链各自形成一个子图，所以 27 个操作最终形成 9 个子图；链间共享输入不影响这个划分。':'8 条 ADDS → CAST 链形成 8 个子图，4 对 UB_COPY_L1 共享写者形成 4 个子图，两个 CONV 各自一个子图：8 + 4 + 2 = 14。'}</div><p class="detail-description">${caseIndex===0?'九个子图之间没有计算先后依赖，但仍共享输入张量。':'输入链在共享写者处汇合，卷积又读取多个输入子图；这些分支／汇合点不满足单出边、单入边条件，因此不会继续合并。'}</p><p class="micro">这些是初始化使用的计算子图，之后的搜索仍允许重新划分。子图划分完成后才开始给每个子图分配核心。</p><button id="go-initial" class="primary next-chapter">用这些子图构造初始解 →</button>`;
  return `<div class="detail-eyebrow">00 / DAG TO GROUPS · ${step+1} OF ${m.grouping.length}</div><h3>${groupingTitles[e.kind]}</h3>${content}<div class="group-count">${e.groups.length}<span>当前子图数量 / ${m.compute_ids.length} 个计算操作</span></div>${sourceRef('graph.cpp:304','build_groups()')}`;
}
function initDetail() {
  const m=model(),c=m.construction;
  if(step===0) return `<div class="detail-eyebrow">01 / INITIALIZATION</div><h3>先划分子图，再决定放在哪个核。</h3><p class="detail-description">${m.compute_ids.length} 个计算操作形成 <strong>${m.initial.groups.length} 个初始子图</strong>。共享写者必须绑定，收缩成环时继续合并，最后合并最多 64 个操作的短链。</p><div class="init-flow"><span>约束：共享写者同核</span><i>→</i><span>约束：禁止成环</span><i>→</i><span>短链合并</span></div><div class="group-table"><table><thead><tr><th>子图</th><th>子图内操作</th><th>工作量</th></tr></thead><tbody>${m.initial.groups.map(g=>`<tr><td>G${g.id}</td><td>${opList(g.ops,'、')}</td><td>${fmt(g.work)}</td></tr>`).join('')}</tbody></table></div><div class="callout">${caseIndex===0?'九条 3 级乘加链分别合并为一个初始子图。行列共享输入是复用机会，不要求所有读取者同核。':'G8～G11 分别绑定两个写入同一张量的 UB_COPY_L1；ADDS → CAST 则合并为短链。共享写者绑定与短链合并的原因不同。'}</div><p class="micro">初始化枚举实际用核数 1～2、三种遍历顺序、三种共享权重及工作量切段，共生成 ${m.candidate_count} 次候选（含重复）。下面回放最终胜出初始解的一条构造路径。</p>${sourceRef('graph.cpp:304','build_groups()')}`;
  if(step<=m.steps.length) {
    const s=m.steps[step-1],g=m.initial.groups[s.group];
    return `<div class="detail-eyebrow">01 / GREEDY · ${step.toString().padStart(2,'0')} OF ${m.steps.length}</div>
      <h3>贪心分核</h3>
      <div class="greedy-score">
        <div class="greedy-score-label">评分公式 · 得分越低越优</div>
        <div class="formula formula-inline" tabindex="0" aria-label="分核评分公式">score = peak + α·end + β·cut + γ·overload</div>
        <div class="weight-guide"><span><b>α</b> 更早完成</span><span><b>β</b> 更少搬运</span><span><b>γ</b> 更均衡负载</span></div>
      </div>
      <p class="detail-description greedy-description">通过拓扑序 / 反向 DFS / 关键路径三种遍历方式取出子图，对允许使用的核心逐一评分，选择得分最低的核心。</p>
      <table class="score-table"><thead><tr><th>候选核</th><th>peak</th><th>end</th><th>cut</th><th>score ↓</th></tr></thead><tbody>${s.choices.map(x=>`<tr class="${x.core===s.chosen?'winner':''}"><td>核 ${x.core}${x.core===s.chosen?' ✓':''}</td><td>${fmt(x.peak,1)}</td><td>${fmt(x.end,1)}</td><td>${fmt(x.cut,1)}</td><td>${fmt(x.score,2)}</td></tr>`).join('')}</tbody></table>
      <div class="callout success greedy-result">本次分配：将子图 G${s.group} 放到 <strong style="color:${colors[s.chosen]}">核 ${s.chosen}</strong>。<div class="greedy-ops">包含 ${opList(g.ops,'、')}</div></div>
      <details class="greedy-help">
        <summary>计算方法与权重含义</summary>
        <dl class="formula-terms"><dt>peak</dt><dd>对该核的每条管道，累加已分配子图的负载与当前子图的负载，再取所有管道中的最大值。</dd><dt>end</dt><dd>先取各前驱的估计完成时间加跨核同步延迟后的最大值（同核不加延迟）；再与当前子图使用的各管道“已有负载 + 当前子图负载”比较，取最大值。</dd><dt>cut</dt><dd>遍历当前子图涉及的张量：外部输入尚未被该核使用时计一次搬运；已有计算写者位于其他核时计换出、换入的双向搬运。将累计字节数除以 DDR 带宽。</dd><dt>overload</dt><dd>取 peak 超过 target 的部分；未超过则为零。</dd><dt>target</dt><dd>先以每个子图最大的管道负载作为其工作量，求和后除以本候选使用的核心数，得到平均目标负载，并设置最小正值下限。</dd></dl><div class="micro formula-weights"><strong>权重的物理含义</strong><p><b>α · 完成时间倾向</b>：越大，越重视前驱等待和管道负载对完成时间的影响，倾向选择估计更早完成的核心。</p><p><b>β · 数据搬运代价</b>：越大，越重视输入复用和减少跨核通信，倾向把能复用数据的子图放到同核。</p><p><b>γ · 负载均衡惩罚</b>：越大，对超过平均目标负载的核心惩罚越强，倾向避免任务过度集中。</p></div>
        <p class="micro">以上为贪心评分；方案的实际完成时间还需要完整模拟复核。</p>
        ${sourceRef('initial_plan.cpp:271','greedy(order, affinity)')}
      </details>`;
  }
  return `<div class="detail-eyebrow">01 / COMPLETE REVIEW</div><h3>完整模拟，选出初始解。</h3><p class="detail-description">逐个子图评分只生成候选。代码将去重后的分核候选全部完整复核，并尝试两种复用／释放排序和局部插入，再按完成时间优先、搬运量次优选出初始解。</p><div class="stats-pair"><div class="good"><small>初始解总完成时间</small><strong>${fmt(m.initial.time)}</strong><small>cycles · C++ 内部模拟</small></div><div><small>实际用核 / 可用核心</small><strong>${usedCores(m.initial.plan)} / 2</strong><small>${fmt(m.initial.copy/1024,2)} KiB 总搬运量</small></div></div><div class="callout success">本例可用核心为核 0、核 1。胜出初始解实际使用 ${usedCores(m.initial.plan)} 核；下方两个核心队列展示最终的初始分配。</div><p class="detail-description">下表记录 ${m.reviews.length} 次完整复核。刚才回放的是最终胜出分核的一次原始贪心构造；本例后续初始化顺序优化没有产生更好的全局初始解。</p><button class="primary next-chapter" id="go-search">以这个初始解开始搜索 →</button>${sourceRef('initial_plan.cpp:614','InitialPlanBuilder::run()')}`;
}

const searchStages = {
  grid_search:{title:'网格分块与重排',category:'数据局部性',description:'将共享输入的输出单元组织成矩形块，比较块形状、行列遍历和分核方式。',source:'search/grid_search.cpp:6'},
  chain_moves:{title:'链迁移',category:'计算均衡',description:'尝试整链或半链迁核，联动共享写者，比较负载改善与新增搬运代价。',source:'search/chain_search.cpp:6'},
  granularity:{title:'子图粒度调整',category:'数据局部性',description:'保持操作序列与核心归属不变，比较连续分组大小；子图边界会影响核内调度。',source:'search/grid_search.cpp:62'},
  producer_search:{title:'生命周期调整',category:'存储驻留与换出',description:'将同核生产者移近消费者，比较缩短输出驻留的收益与延长输入驻留的代价。',source:'search/order_search.cpp:6'}
};
function searchClassification() {
  return `<details class="greedy-help strategy-classification"><summary>策略分类：优化目标 → 具体方法</summary>
    <div class="strategy-category"><h3>计算均衡</h3><h4>链迁移</h4></div>
    <div class="strategy-category"><h3>数据局部性</h3><h4>网格分块与重排 · 子图粒度调整</h4></div>
    <div class="strategy-category"><h3>存储驻留与换出</h3><h4>生命周期调整</h4></div>
    <div class="strategy-category"><h3>依赖与资源等待</h3><h4>分层重排</h4></div>
    <p class="micro">按主要优化目标归类，各方法也可能影响其他目标。子图粒度调整改变分组边界及核内调度，不等同于分层重排。分层重排属于原代码的混合管道排序阶段，此处列明分类，不新增回放步骤。</p>
  </details>`;
}
function searchStatus(e) { return e.status==='pruned'?'下界剪枝':!e.valid?'无效候选':e.accepted?'接受':'保留原解'; }
function searchAction(e) {
  const r=e.recipe;
  if(e.stage==='grid_search')return `${r.rows} × ${r.cols} 分块`;
  if(e.stage==='granularity')return `每 ${e.chunk} 个操作一张子图`;
  if(e.stage==='chain_moves')return ['整链迁移','前半链迁移','后半链迁移','拆链迁移'][r.part];
  return `延后 ${r.moves.length} 个生产者`;
}
function recipeDetail(e) {
  const r=e.recipe;
  if(e.stage==='grid_search')return `<div class="strategy-card"><b>目标分块：${r.rows}×${r.cols} 个输出单元 · 使用 ${r.active} 核</b><p>${r.axis?'按列遍历':'按行遍历'}；${['按输出单元连续区间分核','各块轮流分给核心','将下一块放到已分配单元数最少的核'][r.placement]}。块内先执行同一乘加层的单元，再推进下一层。左侧对应图展示每格包含的 op 和本次实际分块。</p></div>`;
  if(e.stage==='granularity')return `<div class="strategy-card"><b>每核每 ${e.chunk} 个连续操作组成一张子图</b><p>最后不足 ${e.chunk} 个操作的部分单独成图。操作列表与分核不变，比较的是子图边界变化后的完整模拟结果。</p></div>`;
  if(e.stage==='chain_moves') {
    const seedCore=Object.fromEntries(r.seed.flatMap((part,c)=>part.map(op=>[op,c])));
    return `<div class="move-list"><div class="mini-label">实际提案：${searchAction(e)}</div>${r.changes.map(x=>`<div class="move-row"><b>${opLabel(x.op)}</b><span>核 ${seedCore[x.op]} → 核 ${x.core}</span></div>`).join('')}</div>`;
  }
  const line=x=>`<div class="move-row"><b>${opLabel(x.op)}</b><span>核 ${x.core} · 第 ${x.from+1} 位 → 第 ${x.to+1} 位</span></div>`;
  return `<div class="move-list"><div class="mini-label">${searchAction(e)} · 核心归属不变</div>${r.moves.length?r.moves.slice(0,3).map(line).join(''):'<p class="micro">本轮没有生产者满足延后条件。</p>'}${r.moves.length>3?`<details class="more-moves"><summary>其余 ${r.moves.length-3} 次移动</summary>${r.moves.slice(3).map(line).join('')}</details>`:''}</div>`;
}
function originalSearchHelp(e) {
  let text;
  if(e.stage==='grid_search')text='代码枚举固定的 30 种块形状、行／列两种遍历和三种分核方式，并比较活动核心数。候选按序生成，去重、资源下界检查后完整模拟；并非逐个子图尝试所有插入位置。';
  else if(e.stage==='granularity')text='原代码比较的分组大小为 1、2、3、4、6、8、12、16、24、32、48、64、96、128。输入是当前最优方案的实际执行序列；调整分组后，仍由原模拟器决定核内调度。';
  else if(e.stage==='chain_moves')text=`原代码尝试整链、前半链和后半链等提案，先联动共享写者，再以计算负载与必要搬运代价预筛。每类保留至多 8 个提案；${e.recipe.window?'本候选迁核后使用就绪窗口排序，再延后输出膨胀的生产者。':'本候选迁核后按参考执行顺序排列，再尝试延后生产者。'}这里展示的是原代码包含顺序整理的完整候选，不是单独迁核的效果。`;
  else text=`原代码最多进行两轮生产者延后，每轮比较三档驻留代价权重和一个只处理输出膨胀操作的分支。本候选${e.recipe.expanding?'使用输出膨胀分支':`使用驻留代价权重 ${e.recipe.factor}`}；生产者的直接消费者必须都在同核。位置按操作计数，从 1 开始，是每次移动发生时的位置。`;
  return `<details class="greedy-help search-help"><summary>原代码如何构造和接受候选</summary>
    <p class="micro">${text}</p>
    <p class="micro">原搜索接受规则：完成时间下降超过 0.5 cycles；或时间相差不超过 0.5 cycles，且搬运量下降超过 0.5 bytes。候选通过完整模拟后才可能更新最优方案。</p>
    <p class="micro">同一批候选可能从同一个起点生成；评估时逐个与当时最优解比较。移动记录说明候选如何生成，前后切换比较当前最优解和候选。</p>
    <p class="micro">${e.stage==='granularity'?'本步改变子图粒度。':'本候选按原搜索实现以单操作子图提交复核，分组边界与顺序变化也会影响结果。'}${e.valid?`候选总搬运量 ${fmt(e.copy/1024,2)} KiB；当前最优 ${fmt(e.before_copy/1024,2)} KiB。`:''}</p>
    ${sourceRef(searchStages[e.stage].source,e.stage+'()')}
  </details>`;
}
function searchDetail() {
  const m=model(),e=activeSearch();
  if(step===0)return `<div class="detail-eyebrow">02 / ORIGINAL SEARCH TRACE</div><h3>局部搜索</h3>
    ${m.search_scope.map(key=>`<div class="strategy-card strategy-overview"><h3>${searchStages[key].category}</h3><h4 class="search-method">${searchStages[key].title}</h4><p>${searchStages[key].description}</p></div>`).join('')}
    <div class="init-flow"><span>原策略生成候选</span><i>→</i><span>预筛 / 完整模拟</span><i>→</i><span>接受或保留</span></div>
    <p class="micro">从 ${fmt(m.initial.time)} cycles 的初始解出发，默认回放 ${m.focus.length} 个关键候选。数据来自原完整搜索，页面只展开这两个阶段；省略的主起点阶段没有更新最优解。</p>${searchClassification()}`;
  if(!e) {
    const gain=(m.initial.time-m.final.time)/m.initial.time*100;
    return `<div class="detail-eyebrow">02 / ORIGINAL SOLVER RESULT</div><h3>原代码结果：改善 ${fmt(gain,2)}%。</h3>
      <div class="stats-pair"><div><small>初始解 / cycles</small><strong>${fmt(m.initial.time)}</strong></div><div class="good"><small>最终结果 / cycles</small><strong>${fmt(m.final.time)}</strong></div></div>
      <div class="callout success">${caseIndex===0?'网格分块与重排产生了改进；随后链迁移的候选均未进一步改善，保留分块后的最优方案。':'子图粒度调整先产生改进，再通过生产者延后进一步降低完成时间。'}</div>
      <p class="detail-description">总搬运量 ${fmt(m.initial.copy/1024,2)} → ${fmt(m.final.copy/1024,2)} KiB。</p>
      <p class="micro">最终方案已与未修改的原求解器核对一致。只简化讲解范围，不改变策略、调用顺序或接受规则；所有已接受的主起点变化都包含在回放中。</p>`;
  }
  const stage=searchStages[e.stage],prune=e.status==='pruned';
  const reason=prune?'资源下界已超过当前最好完成时间及容差，原代码提前剪枝，未运行完整模拟。':!e.valid?'未得到有效模拟结果，保留原解。':e.accepted?(e.time<e.before-.5?`完成时间减少 ${fmt(e.before-e.time)} cycles，接受候选。`:'完成时间相当但搬运量减少，接受候选。'):Math.abs(e.time-e.before)<=.5?'完成时间相当，搬运量也未达到接受条件，保留原解。':'完整模拟未优于当前最好方案，保留原解。';
  return `<div class="detail-eyebrow">原运行候选 ${String(e.trace_index+1).padStart(3,'0')} <span class="badge ${prune?'prune':e.accepted?'accept':'reject'}">${searchStatus(e)}</span></div>
    <h3>${stage.category}</h3><h4 class="search-method">${stage.title}</h4>${recipeDetail(e)}
    ${e.valid?stats(e.before,e.time):`<div class="callout">当前最好 ${fmt(e.before)} cycles<br>候选：${prune?'下界剪枝，未模拟':'无有效结果'}</div>`}
    <div class="search-decision ${e.accepted?'accepted':''}">${reason}</div>
    <div class="view-switch"><button data-view="before" class="${showCandidate?'':'active'}">比较前的最优方案</button><button data-view="candidate" class="${showCandidate?'active':''}">本次候选</button></div>
    ${originalSearchHelp(e)}`;
}

function renderEvidence() {
  const m=model();
  if(tab===-1){
    $('evidence-kicker').textContent='GROUPING TRACE';$('evidence-title').textContent='从 DAG 到初始子图：完整步骤';
    $('evidence-controls').innerHTML='<span>点击一步，查看对应依赖与合并原因</span>';
    $('evidence-body').innerHTML=`<table><thead><tr><th>回放序号</th><th>处理步骤</th><th>涉及操作</th><th>子图数量变化</th></tr></thead><tbody>${m.grouping.map((e,i)=>`<tr class="clickable${i===step?' selected':''}" data-group-step="${i}" tabindex="0" role="button" aria-label="回放子图划分步骤 ${i+1}"><td>${i+1}</td><td>${groupingTitles[e.kind]}</td><td>${e.focus?opList(e.focus):'—'}</td><td>${e.before?e.before.length+' → ':''}${e.groups.length}</td></tr>`).join('')}</tbody></table>`;
    $('evidence-body').querySelectorAll('[data-group-step]').forEach(el=>{
      const select=()=>{stop();setStep(Number(el.dataset.groupStep??el.dataset['group-step']));};
      el.addEventListener('click',select);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
    });return;
  }
  if(tab===0) {
    $('evidence-kicker').textContent='INITIAL CANDIDATES';
    $('evidence-title').textContent='初始候选的完整复核';
    $('evidence-controls').innerHTML=`<span>${m.candidate_count} 次生成 · ${m.reviews.length} 次完整复核</span>`;
    const best=m.initial;
    $('evidence-body').innerHTML=`<p class="table-note">生成阶段按分核与顺序去重；之后还尝试复用优先、释放优先和局部插入重排。表中为实际复核顺序，标记所有与最终初始解相同的方案。</p><table><thead><tr><th>#</th><th>候选核数 / 实际使用</th><th>构造来源</th><th>各子图核心归属</th><th>粗估 cycles</th><th>完整模拟 cycles</th><th>总搬运 KiB</th><th>结果</th></tr></thead><tbody>${m.reviews.map((r,i)=>{
      const winner=JSON.stringify(r.core)===JSON.stringify(best.core)&&JSON.stringify(r.order)===JSON.stringify(best.order);
      const o=r.origin;
      const origin=!o?'分核后顺序优化':o.method==='greedy'?`${modes[o.mode]} · β=${o.affinity}`:o.method==='cuts'?`${modes[o.mode]} · 工作量切段`:'单核备选';
      return `<tr class="${winner?'selected':''}"><td>${String(i+1).padStart(2,'0')}</td><td>${r.active} / ${new Set(r.core).size}</td><td>${origin}</td><td><span class="order-code" title="${r.core.join(', ')}">[${r.core.join(', ')}]</span></td><td>${o?fmt(o.estimate):'—'}</td><td>${r.valid?fmt(r.time):'无效'}</td><td>${r.valid?fmt(r.copy/1024,2):'—'}</td><td class="${winner?'pool-result':''}">${winner?'胜出方案':'已比较'}</td></tr>`;
    }).join('')}</tbody></table>`;
  } else {
    $('evidence-kicker').textContent='SEARCH CANDIDATES';
    $('evidence-title').textContent=caseIndex===0?'网格分块与链迁移：原运行记录':'子图粒度与生命周期：原运行记录';
    $('evidence-controls').innerHTML=`<span>${showAll?'所选阶段全部候选':'关键候选 · 原运行顺序'}</span><label><input type="checkbox" id="all-candidates" ${showAll?'checked':''}> 全部候选</label>`;
    const current=activeSearch();
    $('evidence-body').innerHTML=`<p class="table-note">点击一行回放该候选。每次都与当时保留的最优解比较，时间单位为 cycles；下界剪枝的候选没有完整模拟时间。</p><table><thead><tr><th>实际序号</th><th>优化目标</th><th>具体方法</th><th>实际配置</th><th>当时最优</th><th>候选结果</th><th>总搬运 KiB</th><th>决策</th></tr></thead><tbody>${searches().map((e,i)=>`<tr class="clickable${current===e?' selected':''}" data-search="${i+1}" tabindex="0" role="button" aria-label="回放候选 ${e.index+1}"><td>${String(e.trace_index+1).padStart(3,'0')}</td><td>${searchStages[e.stage].category}</td><td>${searchStages[e.stage].title}</td><td>${searchAction(e)}</td><td>${fmt(e.before)}</td><td>${e.valid?fmt(e.time):'—'}</td><td>${e.valid?fmt(e.copy/1024,2):'—'}</td><td><span class="badge ${e.status==='pruned'?'prune':e.accepted?'accept':'reject'}">${searchStatus(e)}</span></td></tr>`).join('')}</tbody></table>`;
    $('all-candidates').addEventListener('change',e=>{
      stop();const old=activeSearch();showAll=e.target.checked;
      step=old?Math.max(0,searches().indexOf(old)+1):0;render();
    });
    $('evidence-body').querySelectorAll('[data-search]').forEach(el=>{
      const select=()=>{stop();showCandidate=true;setStep(Number(el.dataset.search));};
      el.addEventListener('click',select);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
    });
  }
}

function render() {
  const m=model();
  $('search-overview').hidden=!(tab===1&&step===maxStep());
  $('search-budget-note').textContent=`本例的阶段上限为 ${m.phaseBudget} 次。`;
  $('core-config').textContent=`可用 2 核（核 0、核 1） · 本例初始 / 最终：${usedCores(m.initial.plan)} / ${usedCores(m.final.plan)} 核`;
  document.querySelectorAll('[data-case]').forEach(el=>{el.classList.toggle('active',Number(el.dataset.case)===caseIndex);el.setAttribute('aria-pressed',Number(el.dataset.case)===caseIndex);});
  document.querySelectorAll('[data-tab]').forEach(el=>{el.classList.toggle('active',Number(el.dataset.tab)===tab);el.setAttribute('aria-pressed',Number(el.dataset.tab)===tab);});
  $('case-note').textContent=caseIndex===0?'3 × 3 输出单元，3 级乘加链；保留行列共享输入。':'两个相邻卷积窗口；保留向量处理链、双写者和重叠输入。';
  $('provenance').innerHTML=`<p><strong>${m.id} · ${m.title}</strong>：${m.note}</p><p>原始图有 ${fmt(m.original_ops)} 个操作（含 COPY），本小图包含 ${m.compute_ids.length} 个计算操作和 ${m.boundary.length} 个合成 COPY 边界。操作与张量在页面中分别从 op_1、tensor_1 连续命名，所有视图保持同一对应关系；操作类型、管道、周期和张量位置、大小均来自原图。截断处以 DDR 输入／输出补齐，不是对原图完整计算结果的等价替代。</p><p>初始化、搜索和模拟均调用原代码。独立副本只添加观察日志，完整执行全部原搜索阶段；网页选取矩阵的网格分块与链迁移、卷积的粒度调整与生命周期调整。省略的主起点阶段没有接受更新，最终方案与未修改的原求解器一致。初始解与最终解已通过仓库 Python 场景 B 评估器交叉检查。两个示例均固定提供 2 个核心（核 0、核 1），让两次演示可直接对照。</p>`;
  $('chapter-summary').innerHTML=tab===-1?`<span>计算操作 → 初始子图</span><br><strong>${m.compute_ids.length} → ${m.initial.groups.length}</strong>`:tab===0?`<span>本例初始解</span><br><strong>${fmt(m.initial.time)}</strong> <span>cycles</span>`:`<span>初始解 → 搜索结果</span><br><strong>${fmt(m.initial.time)} <span style="color:var(--dim);font-size:15px">→</span> ${fmt(m.final.time)}</strong>`;
  $('explanation').classList.toggle('greedy-panel',(tab===0&&step>0&&step<=m.steps.length)||tab===1);
  $('detail').innerHTML=tab===-1?groupingDetail():tab===0?initDetail():searchDetail();
  const initial=$('go-initial');if(initial)initial.addEventListener('click',()=>setTab(0));
  const go=$('go-search');if(go)go.addEventListener('click',()=>setTab(1));
  document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>{showCandidate=el.dataset.view==='candidate';render();}));
  $('step-label').textContent=`STEP ${String(step).padStart(2,'0')} / ${String(maxStep()).padStart(2,'0')}`;
  $('scrubber').max=maxStep();$('scrubber').value=step;
  $('prev').disabled=step===0;$('next').disabled=step===maxStep();
  $('play').textContent=timer?'Ⅱ 暂停':step===maxStep()?'▶ 重新演示':'▶ 自动演示';
  renderGridExplainer();renderGraph();renderQueues();renderEvidence();
}

document.querySelectorAll('[data-case]').forEach(el=>el.addEventListener('click',()=>setCase(Number(el.dataset.case))));
document.querySelectorAll('[data-tab]').forEach(el=>el.addEventListener('click',()=>setTab(Number(el.dataset.tab))));
document.querySelector('.brand').addEventListener('click',e=>{e.preventDefault();setTab(-1);});
$('grid-single').addEventListener('click',()=>jumpGridCandidate(false));
$('grid-multi').addEventListener('click',()=>jumpGridCandidate(true));
$('prev').addEventListener('click',()=>{stop();setStep(step-1);});
$('next').addEventListener('click',()=>{stop();showCandidate=true;setStep(step+1);});
$('reset').addEventListener('click',()=>{stop();selectedNode=null;setStep(0);});
$('scrubber').addEventListener('input',e=>{stop();showCandidate=true;setStep(Number(e.target.value));});
$('show-groups').addEventListener('change',renderGraph);
$('view-dag').addEventListener('click',()=>{graphView='dag';renderGraph();});
$('view-grouped').addEventListener('click',()=>{graphView='grouped';renderGraph();});
$('expand-boundaries').addEventListener('change',renderGraph);
function zoomDag(value){dagZoom=Math.max(.15,Math.min(2,value));renderGraph();}
$('dag-plus').addEventListener('click',()=>zoomDag(dagZoom+.15));
$('dag-minus').addEventListener('click',()=>zoomDag(dagZoom-.15));
$('dag-reset').addEventListener('click',()=>zoomDag(1));
$('dag-fit').addEventListener('click',()=>{const d=dagData(model(),$('expand-boundaries').checked);zoomDag((($('graph-wrap').clientWidth||780)-30)/d.width);});
$('provenance-button').addEventListener('click',()=>{
  const open=$('provenance').hidden;$('provenance').hidden=!open;
  $('provenance-button').setAttribute('aria-expanded',open);$('provenance-button').textContent=open?'收起说明 ↗':'查看缩图说明 ↗';
});
$('play').addEventListener('click',()=>{
  if(timer){stop();return;}
  if(step===maxStep()){step=0;graphView=tab<=0?'dag':'grouped';}
  timer=setInterval(()=>{showCandidate=true;setStep(step+1);if(step>=maxStep()){stop();render();}},3500);
  render();
});
document.addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;
  if(e.key==='ArrowRight'){e.preventDefault();stop();setStep(step+1);}
  if(e.key==='ArrowLeft'){e.preventDefault();stop();setStep(step-1);}
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
render();
