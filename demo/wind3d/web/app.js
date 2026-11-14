import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
const $ = id => document.getElementById(id);
const specs = [ ['speed','来流速度',1,30,'m/s',12], ['angle','迎角',-20,25,'°',5],
  ['flap','襟翼',0,40,'°',0], ['aileron','副翼（差动）',-25,25,'°',0],
  ['elevator','升降舵',-25,25,'°',0], ['rudder','方向舵',-25,25,'°',0] ];
let desired = null, packetMeta = null, commands = Promise.resolve(), pending = 0, debounce;
let frameTag = '', meshID = '', lastStatus = null, transferred = 0;
function showError(message) { $('error').textContent=message; $('error').hidden=!message; }
for (const [key,name,min,max,unit,value] of specs) {
  const section=document.createElement('div'); section.className='control';
  section.innerHTML=`<div class="control-head"><label for="input-${key}">${name}</label><output id="value-${key}">${value.toFixed(1)} ${unit}</output></div><input id="input-${key}" type="range" min="${min}" max="${max}" step="0.5" value="${value}"><div class="control-limits"><span>${min} ${unit}</span><span>${max} ${unit}</span></div>`;
  $('controls').append(section);
  $(`input-${key}`).addEventListener('input',()=> {
    $(`value-${key}`).textContent=`${Number($(`input-${key}`).value).toFixed(1)} ${unit}`;
    $('ack').textContent='等待发送'; clearTimeout(debounce);
    debounce=setTimeout(()=>send({controls:readControls()}),250);
  });
}
function readControls(){return Object.fromEntries(specs.map(([k])=>[k,Number($(`input-${k}`).value)]));}
function setControls(values){for(const [k,,,,unit] of specs){$(`input-${k}`).value=values[k];$(`value-${k}`).textContent=`${Number(values[k]).toFixed(1)} ${unit}`;}}
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
const camera = new THREE.PerspectiveCamera(43,1,0.05,150);
let renderer;
try {renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true,alpha:true});}
catch(error){showError(`WebGL 不可用：${error.message}`);throw error;}
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.minDistance=5;orbit.maxDistance=40;
function home(){camera.position.set(10,6.5,12);orbit.target.set(1,0,0);orbit.update();} home();$('home').onclick=home;
scene.add(new THREE.HemisphereLight(0xc9e7ff,0x475365,2));
const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(-3,8,6);scene.add(light);
const aircraft=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial({vertexColors:true,roughness:.66,metalness:.18,side:THREE.DoubleSide}));scene.add(aircraft);
const flow=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.72}));scene.add(flow);
const loads=new THREE.Group();scene.add(loads);loads.visible=false;
const origin=new THREE.Vector3(0,2,0);
const forceArrow=new THREE.ArrowHelper(new THREE.Vector3(0,1,0),origin,0,0x18e8bd);
const momentArrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),origin,0,0xff55c8);
loads.add(forceArrow,momentArrow);
loads.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),origin]),new THREE.LineBasicMaterial({color:0x566c80})));
$('loads').onchange=()=>loads.visible=$('loads').checked;
$('aircraft').onchange=()=>aircraft.visible=$('aircraft').checked;
$('flow').onchange=()=>flow.geometry.setDrawRange(0,$('flow').checked?Infinity:70);
function arrow(object,values){const v=new THREE.Vector3(...values),length=v.length();object.visible=length>1e-6;if(object.visible){object.setDirection(v.normalize());const size=3*length/(100+length);object.setLength(size,Math.min(.28,size*.4),Math.min(.2,size*.3));}}
function geometry(array,stride,normal){
  const result=new THREE.BufferGeometry(), buffer=new THREE.InterleavedBuffer(array,stride);
  result.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));
  if(normal)result.setAttribute('normal',new THREE.InterleavedBufferAttribute(buffer,3,3));
  result.setAttribute('color',new THREE.InterleavedBufferAttribute(buffer,3,normal?6:3));
  result.computeBoundingSphere(); return result;
}
new ResizeObserver(()=>{const box=$('viewport');renderer.setSize(box.clientWidth,box.clientHeight,false);camera.aspect=box.clientWidth/box.clientHeight;camera.updateProjectionMatrix();}).observe($('viewport'));
function render(){requestAnimationFrame(render);orbit.update();renderer.render(scene,camera);}render();
function updateNumbers(meta,bytes){
  $('hardware').textContent=`单节点 · ${meta.gpus} GPU`;
  $('step').textContent=meta.step.toLocaleString();$('time').textContent=`${meta.time.toFixed(3)} s`;
  $('step-ms').textContent=`${(meta.step_seconds*1000).toFixed(1)} ms`;
  $('residual').textContent=meta.pressure_residual.toExponential(2);$('residual').style.color=meta.pressure_converged?'':'#f6ad6b';
  $('divergence').textContent=meta.divergence.toExponential(2);$('grid').textContent=`${meta.grid} × ${meta.grid/2} × ${meta.grid*3/4}`;
  for(const [key,unit] of [['force','N'],['moment','N·m']]){
    $(key+'-norm').textContent=`${Math.hypot(...meta[key]).toFixed(2)} ${unit}`;
    $(key).textContent=meta[key].map((v,i)=>`${['X','Y','Z'][i]} ${v.toFixed(3)}`).join(' · ');
  }
  const c=meta.controls;
  $('applied').textContent=`已生效：${c.speed.toFixed(1)} m/s · 迎角 ${c.angle.toFixed(1)}° · 襟翼 ${c.flap.toFixed(1)}° · 副翼 ${c.aileron.toFixed(1)}° · 升降 ${c.elevator.toFixed(1)}° · 方向 ${c.rudder.toFixed(1)}°`;
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
    if(meta.version!==1||buffer.byteLength!==4+length+meta.line_vertices*24)throw new Error('显示几何长度无效');
    let meshGeometry=null;
    if(meshID!==meta.mesh){
      const meshResponse=await fetch('/api/mesh/'+meta.mesh);if(!meshResponse.ok)throw new Error('飞机构型暂未同步，正在重试');
      const meshBuffer=await meshResponse.arrayBuffer();if(meshBuffer.byteLength!==meta.mesh_vertices*36)throw new Error('飞机网格长度无效');
      transferred+=Number(meshResponse.headers.get('Content-Length'))||meshBuffer.byteLength;meshGeometry=geometry(new Float32Array(meshBuffer),9,true);
    }
    const lines=geometry(new Float32Array(buffer,4+length),6,false);
    if(meshGeometry){aircraft.geometry.dispose();aircraft.geometry=meshGeometry;meshID=meta.mesh;}
    flow.geometry.dispose();flow.geometry=lines;flow.geometry.setDrawRange(0,$('flow').checked?Infinity:70);
    arrow(forceArrow,meta.force);arrow(momentArrow,meta.moment);loads.visible=$('loads').checked;
    packetMeta=meta;transferred+=Number(response.headers.get('Content-Length'))||buffer.byteLength;frameTag=response.headers.get('ETag');
    updateNumbers(meta,buffer.byteLength);showError('');
    window.wind3dState={meta,meshVertices:aircraft.geometry.attributes.position.count,lineVertices:lines.attributes.position.count};
  } catch(error){showError(error.message);}
  finally {setTimeout(getFrame,350);}
}
async function status(){
  try {
    const response=await fetch('/api/status');if(!response.ok)throw new Error('状态连接失败');
    const state=await response.json();lastStatus=state;
    if(!desired){desired=state;setControls(state.controls);$('pause').textContent=state.paused?'继续':'暂停';}
    else if(!pending&&state.sequence>desired.sequence){desired=state;setControls(state.controls);$('pause').textContent=state.paused?'继续':'暂停';}
    const stale=state.age===null||state.age>5||!!state.error;
    $('connection').textContent=state.error?'GPU 节点异常':state.age===null?'等待 GPU 节点':stale?'显示数据已过期':packetMeta?.paused?'已暂停 · GPU 在线':'实时连接 · GPU 在线';
    $('connection').className='badge '+(stale?'stale':'live');
    $('freshness').textContent=state.age===null?'等待 GPU 节点发布首帧；不会显示模拟数据。':`显示数据更新于 ${state.age.toFixed(1)} 秒前；仅传输飞机网格、流线和诊断。`;
    if(!pending)$('ack').textContent=packetMeta?.ack===desired.sequence?'参数已生效':desired.paused?'待生效（可继续或重置）':'参数待生效';
    if(state.error)showError(state.error);
  }catch(error){$('connection').textContent='登录节点连接中断';$('connection').className='badge stale';showError(error.message);}
  finally{setTimeout(status,1000);}
}
status();getFrame();
