import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {aircraftMaterial,streamMesh,setStreams,volumeRenderer} from './presentation.js';
import {applyHistoryView,onHistoryEdit} from './charts.js';
const $ = id => document.getElementById(id);
$('hardware').textContent='等待共享仿真';
document.querySelector('.view-foot').firstChild.textContent='拖拽旋转 · 滚轮缩放 · 相机与桌面同步 ';
for(const hint of document.querySelectorAll('.hint'))if(hint.textContent.includes('13ea18de'))hint.textContent='载荷箭头采用非线性长度缩放；数值、坐标基准与桌面共享。';
for(const span of $('density').parentElement.querySelectorAll('.control-limits span'))span.textContent=span===span.parentElement.firstElementChild?'稀疏':'密集';
const specs = [ ['speed','来流速度',1,30,'m/s',30], ['angle','机身迎角（相对来流）',-20,25,'°',4],
  ['flap','襟翼',0,40,'°',0], ['aileron','副翼（差动）',-25,25,'°',0],
  ['elevator','升降舵',-25,25,'°',0], ['rudder','方向舵',-25,25,'°',0],
  ['wing_incidence','主翼根部安装角（相对机身，翼尖洗出 3°）',0,6,'°',4.7],
  ['tail_incidence','平尾安装角（整体配平）',-10,10,'°',0], ['winglets','上翘小翼',0,1,'',1],
  ['linked_flaps','内外襟翼联动',0,1,'',1], ['outer_flap','外段襟翼',0,40,'°',0],
  ['cruise_camber','巡航弯度偏角',-5,5,'°',0], ['aileron_droop','副翼对称下偏',0,15,'°',0],
  ['spoiler','对称扰流板',0,60,'°',0], ['spoiler_roll','扰流板差动',-30,30,'°',0],
  ['linked_spoiler_roll','扰流板联动副翼',0,1,'',0], ['slat','前缘缝翼',0,30,'°',0],
  ['linked_slats','缝翼联动襟翼',0,1,'',0] ];
const flags = new Set(['winglets','linked_flaps','linked_spoiler_roll','linked_slats']);
let desired = null, packetMeta = null, commands = Promise.resolve(), pending = 0, debounce;
let densityDebounce,interacting=false;
document.addEventListener('pointerdown',()=>{interacting=true;});
document.addEventListener('pointerup',()=>{interacting=false;});
const viewSpecs=[['width','流线宽度 px',.75,4,.05],['transparent_curvature','完全透明曲率 1/m',0,9.999,.001],
  ['opaque_curvature','完全不透明曲率 1/m',.001,10,.001],['volume','涡量体渲染',0,1,1],
  ['colored_volume','按涡量着色',0,1,1],['volume_density','体密度',.01,.6,.005],
  ['volume_threshold','涡量阈值 1/s',0,60,.1],['volume_step','采样步长 / 网格',.5,2,.05],['history','载荷时间曲线',0,1,1]];
const viewPanel=document.createElement('section');viewPanel.className='panel';
viewPanel.innerHTML='<h2>共享显示设置</h2><p class="hint">与桌面使用同一仿真、原点和显示参数。</p>';
document.querySelector('aside').append(viewPanel);
for(const [key,label,min,max,step] of viewSpecs){
  const row=document.createElement('div');row.className='control';
  row.innerHTML=`<label for="view-${key}">${label} <output id="view-value-${key}"></output></label><input id="view-${key}" type="${max===1?'checkbox':'range'}" min="${min}" max="${max}" step="${step}">`;
  viewPanel.append(row);
  $('view-'+key).onchange=()=>{const input=$('view-'+key);const value=input.type==='checkbox'?Number(input.checked):Number(input.value);sendView({[key]:value});};
}
function setDensity(d){$('density').value=d;$('density-value').textContent=`${d} 级 · 全空间自适应占位播种`;}
$('density').oninput=()=>{const d=Number($('density').value);setDensity(d);clearTimeout(densityDebounce);densityDebounce=setTimeout(()=>send({streamline_density:d}),250);};
let frameTag = '', meshID = '', lastStatus = null, transferred = 0;
function showError(message) { $('error').textContent=message; $('error').hidden=!message; }
for (const [key,name,min,max,unit,value] of specs) {
  const section=document.createElement('div'); section.className='control';
  section.innerHTML=`<div class="control-head"><label for="input-${key}">${name}</label><output id="value-${key}">${value.toFixed(1)} ${unit}</output></div><input id="input-${key}" type="${flags.has(key)?'checkbox':'range'}" min="${min}" max="${max}" step="${key==='tail_incidence'?'0.001':'0.01'}" value="${value}"><div class="control-limits" ${flags.has(key)?'hidden':''}><span>${min} ${unit}</span><span>${max} ${unit}</span></div>`;
  $('controls').append(section);
  if(flags.has(key)) $(`input-${key}`).checked=Boolean(value);
  $(`input-${key}`).addEventListener('input',()=> {
    updateControlLabels();
    $('ack').textContent='等待发送'; clearTimeout(debounce);
    debounce=setTimeout(()=>send({controls:readControls()}),250);
  });
}
function readControls(){return Object.fromEntries(specs.map(([k])=>[k,flags.has(k)?Number($(`input-${k}`).checked):Number($(`input-${k}`).value)]));}
function updateControlLabels(){
  for(const [k,,,,unit] of specs){const input=$(`input-${k}`);$(`value-${k}`).textContent=flags.has(k)?(input.checked?'开启':'关闭'):`${Number(input.value).toFixed(k==='tail_incidence'?3:2)} ${unit}`;}
  $('input-outer_flap').disabled=$('input-linked_flaps').checked;
  $('input-spoiler_roll').disabled=$('input-linked_spoiler_roll').checked;
  $('input-slat').disabled=$('input-linked_slats').checked;
}
function setControls(values){for(const [k,,,,,fallback] of specs){const value=values[k]??fallback;const input=$(`input-${k}`);if(flags.has(k)) input.checked=Boolean(value);else input.value=value;}updateControlLabels();}
updateControlLabels();
const cgReadout=document.createElement('p');cgReadout.className='hint';
$('moment').closest('.panel').append(cgReadout);
function send(patch){
  ++pending; $('ack').textContent='发送中…';
  commands=commands.catch(()=>{}).then(async()=>{
    const response=await fetch('/api/controls',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    const result=await response.json(); if(!response.ok) throw new Error(result.error);
    desired=result; $('pause').textContent=desired.paused?'继续':'暂停'; showError('');
  }).catch(error=>showError(`控制发送失败：${error.message}`)).finally(()=>--pending);
}
$('pause').onclick=()=>{clearTimeout(debounce);send({controls:readControls(),paused:!desired?.paused});};
$('reset').onclick=()=>{clearTimeout(debounce);send({controls:readControls(),reset:true});};
$('defaults').onclick=()=>{clearTimeout(debounce);const controls=Object.fromEntries(specs.map(s=>[s[0],s[5]]));setControls(controls);send({controls,reset:true});};

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45,1,0.1,80);
let renderer;
try {renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true,alpha:true});}
catch(error){showError(`WebGL 不可用：${error.message}`);throw error;}
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.setClearColor(new THREE.Color(.012,.02,.035),1);
const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=false;orbit.minDistance=.6;orbit.maxDistance=40;
function home(){camera.position.set(-7.272,4.089,6.388);orbit.target.set(0,0,0);orbit.update();} home();$('home').onclick=()=>{home();if(packetMeta?.version===2)sendView({yaw:-.85,pitch:.4,distance:10.5});};
scene.add(new THREE.HemisphereLight(0xc9e7ff,0x475365,2));
const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(-3,8,6);scene.add(light);
const aircraft=new THREE.Mesh(new THREE.BufferGeometry(),aircraftMaterial());scene.add(aircraft);
const flow=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.72}));scene.add(flow);
const streams=streamMesh();scene.add(streams);
const composite=volumeRenderer(renderer,scene,camera,streams);
let viewSettings=null,volumeID='',applyingView=false,orbitActive=false;
function sendView(patch){send({view:patch});}
onHistoryEdit(sendView);
function applyView(v){
  if(!v)return;viewSettings=v;applyingView=true;
  for(const key of ['flow','loads','aircraft','domain']){$(key).checked=!!v[key];}
  aircraft.visible=!!v.aircraft;loads.visible=!!v.loads;domain.visible=!!v.domain;streams.visible=!!v.flow;
  streams.material.uniforms.width.value=v.width;streams.material.uniforms.lower.value=v.transparent_curvature;
  streams.material.uniforms.upper.value=v.opaque_curvature;
  if(!orbitActive&&!pending)camera.position.set(v.distance*Math.cos(v.pitch)*Math.sin(v.yaw),v.distance*Math.sin(v.pitch),v.distance*Math.cos(v.pitch)*Math.cos(v.yaw));
  orbit.target.set(0,0,0);orbit.update();applyingView=false;
  if(!interacting)for(const [key] of viewSpecs){const input=$('view-'+key);if(input.type==='checkbox')input.checked=!!v[key];else input.value=v[key];$('view-value-'+key).textContent=Number(v[key]).toFixed(3);}
  if(v.history_window!==undefined)applyHistoryView(v);
}
orbit.enablePan=false;
orbit.addEventListener('start',()=>{orbitActive=true;});
orbit.addEventListener('end',()=>{orbitActive=false;if(!applyingView&&packetMeta?.version===2){const p=camera.position,d=p.length();sendView({yaw:Math.atan2(p.x,p.z),pitch:Math.asin(p.y/d),distance:d});}});
const loads=new THREE.Group();scene.add(loads);loads.visible=false;
const origin=new THREE.Vector3(0,2,0);
const forceArrow=new THREE.ArrowHelper(new THREE.Vector3(0,1,0),origin,0,0x18e8bd);
const momentArrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),origin,0,0xff55c8);
loads.add(forceArrow,momentArrow);
loads.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),origin]),new THREE.LineBasicMaterial({color:0x566c80})));
$('loads').onchange=()=>{loads.visible=$('loads').checked;if(packetMeta?.version===2)sendView({loads:Number(loads.visible)});};
$('aircraft').onchange=()=>{aircraft.visible=$('aircraft').checked;if(packetMeta?.version===2)sendView({aircraft:Number(aircraft.visible)});};
const domain=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x7fc6ff,transparent:true,opacity:.8}));scene.add(domain);domain.visible=false;
let domainKey='';
function updateDomain(meta){
  if(!meta.domain_min||!meta.domain_max)return;
  const key=JSON.stringify([meta.domain_min,meta.domain_max]);
  if(key!==domainKey){
    const a=new THREE.Vector3(...meta.domain_min),b=new THREE.Vector3(...meta.domain_max),size=b.clone().sub(a);
    const box=new THREE.BoxGeometry(size.x,size.y,size.z);domain.geometry.dispose();domain.geometry=new THREE.EdgesGeometry(box);box.dispose();
    domain.position.copy(a.add(b).multiplyScalar(.5));domainKey=key;
  }
  $('domain-size').textContent=`计算域 ${meta.domain_max.map((v,i)=>v-meta.domain_min[i]).join(' × ')} m · 网格间距 ${(16/meta.grid).toFixed(4)} m`;
}
$('domain').onchange=()=>{
  domain.visible=$('domain').checked;
  if(packetMeta?.version===2){sendView({domain:Number(domain.visible)});return;}
  if(domain.visible&&domainKey){
    domain.geometry.computeBoundingSphere();const radius=domain.geometry.boundingSphere.radius;
    const halfFov=Math.min(camera.fov*Math.PI/360,Math.atan(Math.tan(camera.fov*Math.PI/360)*camera.aspect));
    const distance=radius/Math.sin(halfFov)*1.05,direction=camera.position.clone().sub(orbit.target).normalize();
    orbit.maxDistance=Math.max(40,distance*1.2);orbit.target.copy(domain.position);camera.position.copy(domain.position).addScaledVector(direction,distance);orbit.update();
  }
};
$('flow').onchange=()=>{if(packetMeta?.version===2){streams.visible=$('flow').checked;sendView({flow:Number(streams.visible)});}else flow.geometry.setDrawRange(0,$('flow').checked?Infinity:70);};
function arrow(object,values){const v=new THREE.Vector3(...values),length=v.length();object.visible=length>1e-6;if(object.visible){object.setDirection(v.normalize());const size=3*length/(100+length);object.setLength(size,Math.min(.28,size*.4),Math.min(.2,size*.3));}}
function geometry(array,stride,normal){
  const result=new THREE.BufferGeometry(), buffer=new THREE.InterleavedBuffer(array,stride);
  result.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));
  if(normal)result.setAttribute('normal',new THREE.InterleavedBufferAttribute(buffer,3,3));
  result.setAttribute('color',new THREE.InterleavedBufferAttribute(buffer,3,normal?6:3));
  result.computeBoundingSphere(); return result;
}
new ResizeObserver(()=>{const box=$('viewport');renderer.setSize(box.clientWidth,box.clientHeight,false);camera.aspect=box.clientWidth/box.clientHeight;camera.updateProjectionMatrix();}).observe($('viewport'));
let modelDimensions=null, previousScale='';
function measureModel(mesh,angle){
  // Undo incidence so pitch does not change the model's longitudinal length.
  const positions=mesh.attributes.position,a=angle*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
  const minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i<positions.count;i++){
    const x=positions.getX(i),y=positions.getY(i),p=[c*x-s*y,s*x+c*y,positions.getZ(i)];
    for(let k=0;k<3;k++){minimum[k]=Math.min(minimum[k],p[k]);maximum[k]=Math.max(maximum[k],p[k]);}
  }
  modelDimensions={length:maximum[0]-minimum[0],span:maximum[2]-minimum[2],height:maximum[1]-minimum[1]};
  $('model-size').textContent=`模型长度 ${modelDimensions.length.toFixed(2)} m · 翼展 ${modelDimensions.span.toFixed(2)} m`;
}
function updateScale(){
  const height=renderer.domElement.clientHeight;
  if(!height)return;
  const metersPerPixel=2*camera.position.distanceTo(orbit.target)*Math.tan(camera.fov*Math.PI/360)/height;
  const target=120*metersPerPixel,base=10**Math.floor(Math.log10(target)),ratio=target/base;
  const meters=(ratio>=5?5:ratio>=2?2:1)*base,pixels=meters/metersPerPixel;
  const key=`${meters}-${pixels.toFixed(1)}`;
  if(key!==previousScale){
    $('scale-bar').style.width=`${pixels.toFixed(1)}px`;
    $('scale-label').textContent=`${Number(meters.toPrecision(3))} m`;
    $('scale-bar').dataset.meters=meters;$('scale-bar').dataset.metersPerPixel=metersPerPixel;
    previousScale=key;
  }
}
function render(){requestAnimationFrame(render);orbit.update();updateScale();composite.render(viewSettings);}render();
function updateNumbers(meta,bytes){
  $('hardware').textContent=meta.run?`共享运行 ${meta.run}`:`单节点 · ${meta.gpus??1} GPU`;
  $('step').textContent=meta.step.toLocaleString();$('time').textContent=`${meta.time.toFixed(3)} s`;
  $('step-ms').textContent=`${(meta.step_seconds*1000).toFixed(1)} ms`;
  $('residual').textContent=meta.pressure_residual.toExponential(2);$('residual').style.color=meta.pressure_converged?'':'#f6ad6b';
  $('divergence').textContent=meta.divergence.toExponential(2);$('grid').textContent=`${meta.grid} × ${meta.grid/2} × ${meta.grid*3/4}`;
  for(const [key,unit] of [['force','N'],['moment','N·m']]){
    $(key+'-norm').textContent=`${Math.hypot(...meta[key]).toFixed(2)} ${unit}`;
    $(key).textContent=meta[key].map((v,i)=>`${['X','Y','Z'][i]} ${v.toFixed(3)}`).join(' · ');
  }
  cgReadout.textContent=meta.model_origin_body?
    '力矩统一关于固定模型原点 (0, 0, 0)；质量重心尚未指定。':
    '旧坐标数据：请更新仿真进程后查看新原点力矩。';
  $('mean-ld').textContent=meta.mean_ld == null ? '尚无有效平均值' : `${meta.mean_ld.toFixed(3)}（重置以来 ${meta.average_seconds.toFixed(3)} s 时间加权平均）`;
  updateDomain(meta);
  $('density-applied').textContent=meta.version===2?`全空间流线：${meta.streamline_density} 级 · ${meta.line_vertices/2} 段 · 曲率沿下游累积`:
    `旧版流线：${meta.streamline_density??1} 级`;
  const c=meta.controls;
  const slat=c.linked_slats?0.75*Math.max(c.flap,c.linked_flaps?c.flap:(c.outer_flap??0)):(c.slat??0);
  $('applied').textContent=`已生效：${c.speed.toFixed(1)} m/s · 机身迎角 ${c.angle.toFixed(2)}° · 主翼安装 ${(c.wing_incidence??1.5).toFixed(2)}° · 平尾 ${(c.tail_incidence??0).toFixed(3)}° · 襟翼 ${c.flap.toFixed(1)}° · 缝翼 ${slat.toFixed(1)}° · 扰流板 ${(c.spoiler??0).toFixed(1)}°`;
  $('payload').textContent=`显示帧 ${(bytes/1024).toFixed(0)} KB · 已接收 ${(transferred/1048576).toFixed(1)} MB`;
}
async function getFrame(){
  try {
    const response=await fetch('/api/frame',{headers:frameTag?{'If-None-Match':frameTag}:{}});
    if(response.status===304||response.status===404)return;
    if(!response.ok)throw new Error(`显示帧 HTTP ${response.status}`);
    const buffer=await response.arrayBuffer(), view=new DataView(buffer);
    if(buffer.byteLength<4)throw new Error('显示帧不完整');
    const length=view.getUint32(0,true);if(length>65536||4+length>buffer.byteLength)throw new Error('显示帧头无效');
    const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,length)));
    if(![1,2].includes(meta.version)||buffer.byteLength!==4+length+(meta.line_vertices+(meta.decoration_vertices??0))*(meta.version===2?36:24))throw new Error('显示几何长度无效');
    let meshGeometry=null;
    if(meshID!==meta.mesh){
      $('payload').textContent='正在加载飞机构型…';
      const meshResponse=await fetch('/api/mesh/'+meta.mesh);if(!meshResponse.ok)throw new Error('飞机构型暂未同步，正在重试');
      const meshBuffer=await meshResponse.arrayBuffer();if(meshBuffer.byteLength!==meta.mesh_vertices*36)throw new Error('飞机网格长度无效');
      transferred+=Number(meshResponse.headers.get('Content-Length'))||meshBuffer.byteLength;meshGeometry=geometry(new Float32Array(meshBuffer),9,true);
    }
    let volumeData=null;
    if(meta.version===2&&meta.volume&&meta.volume!==volumeID){
      const response=await fetch('/api/volume/'+meta.volume);if(!response.ok)throw new Error('Volume snapshot unavailable');
      const bytes=await response.arrayBuffer();if(bytes.byteLength!==meta.grid*(meta.grid/2)*(meta.grid*3/4)*4)throw new Error('Invalid volume size');
      volumeData=new Float32Array(bytes);
    }
    const rawLines=new Float32Array(buffer,4+length,meta.line_vertices*(meta.version===2?9:6));
    const lines=meta.version===1?geometry(rawLines,6,false):geometry(new Float32Array(buffer,4+length+meta.line_vertices*36),9,true);
    if(meta.version===2){setStreams(streams,rawLines);flow.visible=true;applyView(meta.view);domain.visible=false;loads.visible=false;
      if(volumeData){composite.set(volumeData,meta.grid);volumeID=meta.volume;}
    }
    if(meshGeometry){aircraft.geometry.dispose();aircraft.geometry=meshGeometry;meshID=meta.mesh;measureModel(meshGeometry,meta.controls.angle);}
    flow.material.opacity=meta.version===2?1:.72/(meta.streamline_density??1);
    flow.geometry.dispose();flow.geometry=lines;flow.geometry.setDrawRange(0,meta.version===2||$('flow').checked?Infinity:70);
    arrow(forceArrow,meta.force);arrow(momentArrow,meta.moment);loads.visible=meta.version===1&&$('loads').checked;
    packetMeta=meta;transferred+=Number(response.headers.get('Content-Length'))||buffer.byteLength;frameTag=response.headers.get('ETag');
    updateNumbers(meta,buffer.byteLength);showError('');
    window.wind3dState={meta,modelDimensions,domainBounds:[meta.domain_min,meta.domain_max],meshVertices:aircraft.geometry.attributes.position.count,lineVertices:meta.line_vertices};
  } catch(error){showError(`显示数据请求失败，正在重试：${error.message}`);}
  finally {setTimeout(getFrame,500);}
}
async function status(){
  try {
    const response=await fetch('/api/status');if(!response.ok)throw new Error('状态连接失败');
    const state=await response.json();lastStatus=state;
    if(!desired){desired=state;setControls(state.controls);setDensity(state.streamline_density??2);$('pause').textContent=state.paused?'继续':'暂停';}
    else if(!pending&&!interacting&&(state.sequence>desired.sequence||packetMeta?.version===2)){desired=state;setControls(state.controls);setDensity(state.streamline_density??2);$('pause').textContent=state.paused?'继续':'暂停';}
    const stale=state.age===null||state.age>5||!!state.error;
    $('connection').textContent=state.error?'GPU 节点异常':state.age===null?'等待 GPU 节点':stale?'显示数据已过期':!packetMeta?'GPU 在线 · 加载画面…':packetMeta?.paused?'已暂停 · GPU 在线':'实时连接 · GPU 在线';
    $('connection').className='badge '+(stale?'stale':'live');
    $('freshness').textContent=state.age===null?'等待仿真发布首帧。':`仿真心跳 ${state.age.toFixed(1)} 秒前；网格、流线、载荷及启用的体数据来自同一仿真。`;
    if(!pending)$('ack').textContent=packetMeta?.ack===desired.sequence?'参数已生效':desired.paused?'待生效（可继续或重置）':'参数待生效';
    if(state.error)showError(state.error);
  }catch(error){$('connection').textContent='登录节点连接中断';$('connection').className='badge stale';showError(error.message);}
  finally{setTimeout(status,1000);}
}
status();getFrame();
