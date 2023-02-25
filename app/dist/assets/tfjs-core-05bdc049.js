import"./long-edff4021.js";import{s as Gn}from"./seedrandom-794c9125.js";/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qs=1e-7,zs=1e-4;class Us{refCount(t){return Y("refCount")}incRef(t){return Y("incRef")}timerAvailable(){return!0}time(t){return Y("time")}read(t){return Y("read")}readSync(t){return Y("readSync")}readToGPU(t,n){return Y("readToGPU")}numDataIds(){return Y("numDataIds")}disposeData(t,n){return Y("disposeData")}write(t,n,r){return Y("write")}move(t,n,r,s,o){return Y("move")}createTensorFromGPUData(t,n,r){return Y("createTensorFromGPUData")}memory(){return Y("memory")}floatPrecision(){return Y("floatPrecision")}epsilon(){return this.floatPrecision()===32?qs:zs}dispose(){return Y("dispose")}}function Y(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dd(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,Ws(e,t,n)}function Ws(e,t,n){const r=e[t];e[t]=e[n],e[n]=r}function f(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function Vs(e,t,n=""){f(pe(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function Re(e){f(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function J(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function pe(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function ae(e){return e%1===0}function ee(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function ht(e,t){const n=t.length;return e=e==null?t.map((r,s)=>s):[].concat(e),f(e.every(r=>r>=-n&&r<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),f(e.every(r=>ae(r)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(r=>r<0?n+r:r)}function Hs(e,t){const n=[],r=[],s=t!=null&&Array.isArray(t)&&t.length===0,o=t==null||s?null:ht(t,e).sort();let i=0;for(let a=0;a<e.length;++a){if(o!=null){if(o[i]===a&&e[a]!==1)throw new Error(`Can't squeeze axis ${a} since its dim '${e[a]}' is not 1`);(o[i]==null||o[i]>a)&&e[a]===1&&(n.push(e[a]),r.push(a)),o[i]<=a&&i++}e[a]!==1&&(n.push(e[a]),r.push(a))}return{newShape:n,keptDims:r}}function js(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function Xs(e,t){for(let n=0;n<e.length;n++){const r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function Ys(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function Te(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function Js(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function Le(e){return typeof e=="string"||e instanceof String}function Zs(e){return typeof e=="boolean"}function Qs(e){return typeof e=="number"}function Ke(e){return Array.isArray(e)?Ke(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":Qs(e)?"float32":Le(e)?"string":Zs(e)?"bool":"float32"}function Ie(e){return!!(e&&e.constructor&&e.call&&e.apply)}function Oe(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function Rn(e,t,n,r=!1){const s=new Array;if(t.length===1){const o=t[0]*(r?2:1);for(let i=0;i<o;i++)s[i]=n[e+i]}else{const o=t[0],i=t.slice(1),a=i.reduce((c,u)=>c*u)*(r?2:1);for(let c=0;c<o;c++)s[c]=Rn(e+c*a,i,n,r)}return s}function un(e,t,n=!1){if(e.length===0)return t[0];const r=e.reduce((s,o)=>s*o)*(n?2:1);if(r===0)return[];if(r!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return Rn(0,e,t,n)}function Ln(e,t){const n=qe(e,t);for(let r=0;r<n.length;r++)n[r]=1;return n}function qe(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function bt(e){e.forEach(t=>{f(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function ze(e){return e&&e.then&&typeof e.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ln="tfjsflags";class to{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=eo,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(F().getBool("IS_TEST")||F().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,r){if(this.flagRegistry[t]={evaluationFn:n,setHook:r},this.urlFlags[t]!=null){const s=this.urlFlags[t];F().getBool("IS_TEST")||F().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${s}.`),this.set(t,s)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(ze(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);ln in t&&t[ln].split(",").forEach(r=>{const[s,o]=r.split(":");this.urlFlags[s]=ro(s,o)})}}function eo(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...r)=>(no(t,r[0],r[1]),r.join("="))),t}function no(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function ro(e,t){if(t=t.toLowerCase(),t==="true"||t==="false")return t==="true";if(`${+t}`===t)return+t;throw new Error(`Could not parse value flag value ${t} for flag ${e}.`)}function F(){return Kn}let Kn=null;function so(e){Kn=e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let we;function On(){if(we==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");we=e}return we}function oo(){const e=On();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function Ue(e,t){const n=oo();if(n.has(e))return n.get(e);{const r=t();return n.set(e,r),n.get(e)}}const qn="Abs",ao="Acos",io="Acosh",We="Add",co="AddN",uo="All",lo="Any",zn="ArgMax",ho="ArgMin",fo="Asin",po="Asinh",go="Atan",mo="Atanh",bo="Atan2",Un="AvgPool",yo="AvgPoolGrad",Wn="AvgPool3D",ko="AvgPool3DGrad",Vn="BatchMatMul",Hn="BatchToSpaceND",wo="Bincount",xo="BroadcastTo",Ve="Cast",$o="Ceil",jn="ClipByValue",vo="Complex",Xn="ComplexAbs",Yn="Concat",Jn="Conv2D",So="Conv2DBackpropFilter",Zn="Conv2DBackpropInput",Qn="Conv3D",Eo="Conv3DBackpropFilterV2",To="Conv3DBackpropInputV2",tr="Cos",er="Cosh",Io="Cumprod",nr="Cumsum",No="CropAndResize",Do="DenseBincount",rr="DepthwiseConv2dNative",Ao="DepthwiseConv2dNativeBackpropFilter",Co="DepthwiseConv2dNativeBackpropInput",Fo="Dilation2D",Bo="Dilation2DBackpropInput",Mo="Dilation2DBackpropFilter",sr="RealDiv",or="Elu",_o="EluGrad",Po="Erf",Go="Equal",ar="Exp",ir="ExpandDims",Ro="Expm1",Lo="Fill",Ko="FlipLeftRight",cr="Floor",ur="FloorDiv",lr="FusedBatchNorm",hr="GatherV2",Oo="Greater",fr="GreaterEqual",He="Identity",qo="Imag",zo="IsFinite",Uo="IsInf",Wo="IsNan",dr="LeakyRelu",Vo="Less",Ho="LessEqual",pr="Log",gr="Log1p",jo="LogicalAnd",Xo="LogicalNot",Yo="LogSoftmax",Jo="LRN",Zo="LRNGrad",mr="Max",br="Maximum",yr="MaxPool",Qo="MaxPoolGrad",kr="MaxPool3D",ta="MaxPool3DGrad",wr="Mean",xr="Min",$r="Minimum",ea="MirrorPad",na="Mod",vr="Multiply",Sr="Neg",ra="NotEqual",sa="NonMaxSuppressionV3",oa="NonMaxSuppressionV4",aa="NonMaxSuppressionV5",Er="OnesLike",Tr="OneHot",Ir="Pack",Nr="PadV2",Dr="Pow",Ar="Prelu",ia="Prod",ca="Range",ua="Real",la="Reciprocal",Cr="Relu",Fr="Reshape",Br="ResizeNearestNeighbor",ha="ResizeNearestNeighborGrad",Mr="ResizeBilinear",fa="ResizeBilinearGrad",_r="Relu6",Pr="Reverse",Gr="Round",Rr="Rsqrt",Lr="Select",Kr="Selu",Or="Slice",qr="Sin",zr="Sinh",da="Sign",Ur="Sigmoid",Wr="Softplus",Vr="Sqrt",Hr="Sum",jr="SpaceToBatchND",Xr="SplitV",Yr="Softmax",pa="SquaredDifference",ga="Square",Jr="Sub",ma="Tan",Zr="Tanh",je="Tile",ba="Transform",ne="Transpose",Qr="Unpack",ts="UnsortedSegmentSum",es="ZerosLike",ns="Step",hn="FromPixels",ya="RotateWithOffset",fn="_FusedMatMul",dn="FusedConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bt(...e){F().getBool("IS_TEST")||F().getBool("PROD")||console.warn(...e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rs=Ue("kernelRegistry",()=>new Map),Ne=Ue("gradRegistry",()=>new Map);function De(e,t){const n=wa(e,t);return rs.get(n)}function pn(e){return Ne.get(e)}function gn(e){const t=rs.entries(),n=[];for(;;){const{done:r,value:s}=t.next();if(r)break;const[o,i]=s,[a]=o.split("_");a===e&&n.push(i)}return n}function ka(e){const{kernelName:t}=e;Ne.has(t)&&F().getBool("DEBUG")&&Bt(`Overriding the gradient for '${t}'`),Ne.set(t,e)}function wa(e,t){return`${t}_${e}`}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xa(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function ss(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=Vt(e)),F().getBool("DEBUG")&&Xs(e,t),xa(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let r=0;r<n.length;++r)Math.round(e[r])!==0&&(n[r]=1);return n}else throw new Error(`Unknown data type ${t}`)}function ie(){return F().platform.now()}function $a(e,t="utf-8"){return t=t||"utf-8",F().platform.encode(e,t)}function mn(e,t="utf-8"){return t=t||"utf-8",F().platform.decode(e,t)}function ot(e){return F().platform.isTypedArray(e)}function Vt(e,t=[],n=!1){if(t==null&&(t=[]),typeof e=="boolean"||typeof e=="number"||typeof e=="string"||ze(e)||e==null||ot(e)&&n)t.push(e);else if(Array.isArray(e)||ot(e))for(let r=0;r<e.length;++r)Vt(e[r],t,n);else{let r=-1;for(const s of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(s)&&(r=Math.max(r,Number(s)));for(let s=0;s<=r;s++)Vt(e[s],t,n)}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class va{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new Ea)}profileKernel(t,n,r){let s;const o=()=>{s=r()};let i;const a=ie();if(this.backendTimer.timerAvailable())i=this.backendTimer.time(o);else{o();for(const u of s)u.dataSync();i=Promise.resolve({kernelMs:ie()-a})}if(F().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let u=0;u<s.length;u++){const h=s[u];h.data().then(l=>{Sa(l,h.dtype,t)})}return{kernelName:t,outputs:s,inputs:n,timeMs:i.then(u=>u.kernelMs),extraInfo:i.then(u=>u.getExtraProfileInfo!=null?u.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:r,timeMs:s,inputs:o,extraInfo:i}=t;r.forEach(a=>{Promise.all([a.data(),s,i]).then(c=>{this.logger.logKernelProfile(n,a,c[0],c[1],o,c[2])})})}}function Sa(e,t,n){if(t!=="float32")return!1;for(let r=0;r<e.length;r++){const s=e[r];if(isNaN(s)||!isFinite(s))return console.warn(`Found ${s} in the result of '${n}'`),!0}return!1}class Ea{logKernelProfile(t,n,r,s,o,i){const a=typeof s=="number"?ee(`${s}ms`,9):s.error,c=ee(t,25),u=n.rank,h=n.size,l=ee(n.shape.toString(),14);let p="";for(const m in o){const k=o[m];if(k!=null){const $=k.shape||n.shape,w=$.length;p+=`${m}: ${w}D ${w>0?$:""} `}}console.log(`%c${c}	%c${a}	%c${u}D ${l}	%c${h}	%c${p}	%c${i}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ta(e,t,n){const r={},s={};for(let c=0;c<t.length;c++)r[t[c].id]=!0;for(let c=0;c<e.length;c++){const u=e[c],h=u.inputs;for(const l in h){const p=h[l];let m=!1;for(let k=0;k<t.length;k++)if(r[p.id]){u.outputs.forEach($=>r[$.id]=!0),m=!0,s[u.id]=!0;break}if(m)break}}const o={};o[n.id]=!0;const i={};for(let c=e.length-1;c>=0;c--){const u=e[c],h=u.inputs;for(let l=0;l<u.outputs.length;l++)if(o[u.outputs[l].id]){for(const p in h)o[h[p].id]=!0,i[u.id]=!0;break}}const a=[];for(let c=0;c<e.length;c++){const u=e[c];if(s[u.id]&&i[u.id]){const h={};for(const p in u.inputs){const m=u.inputs[p];r[m.id]&&(h[p]=m)}const l=Object.assign({},u);l.inputs=h,l.outputs=u.outputs,a.push(l)}}return a}function Ia(e,t,n,r){for(let s=t.length-1;s>=0;s--){const o=t[s],i=[];if(o.outputs.forEach(c=>{const u=e[c.id];u!=null?i.push(u):i.push(null)}),o.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${o.kernelName}.`);const a=o.gradient(i);for(const c in o.inputs){if(!(c in a))throw new Error(`Cannot backprop through input ${c}. Available gradients found: ${Object.keys(a)}.`);const u=n(()=>a[c]());if(u.dtype!=="float32")throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input ${c} must have 'float32' dtype, but has '${u.dtype}'`);const h=o.inputs[c];if(!pe(u.shape,h.shape))throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input '${c}' has shape '${u.shape}', which does not match the shape of the input '${h.shape}'`);if(e[h.id]==null)e[h.id]=u;else{const l=e[h.id];e[h.id]=r(l,u),l.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bn=20,qt=3,xe=7;function Na(e,t,n,r){const s=Oe(t),o=Da(e,t,n,s),i=t.length,a=re(e,t,n,s,o),c=["Tensor"];return r&&(c.push(`  dtype: ${n}`),c.push(`  rank: ${i}`),c.push(`  shape: [${t}]`),c.push("  values:")),c.push(a.map(u=>"    "+u).join(`
`)),c.join(`
`)}function Da(e,t,n,r){const s=J(t),o=r[r.length-1],i=new Array(o).fill(0),a=t.length,c=n==="complex64"?Ut(e):e;if(a>1)for(let u=0;u<s/o;u++){const h=u*o;for(let l=0;l<o;l++)i[l]=Math.max(i[l],zt(c[h+l],0,n).length)}return i}function zt(e,t,n){let r;return Array.isArray(e)?r=`${parseFloat(e[0].toFixed(xe))} + ${parseFloat(e[1].toFixed(xe))}j`:Le(e)?r=`'${e}'`:n==="bool"?r=os(e):r=parseFloat(e.toFixed(xe)).toString(),ee(r,t)}function os(e){return e===0?"false":"true"}function re(e,t,n,r,s,o=!0){const i=n==="complex64"?2:1,a=t[0],c=t.length;if(c===0){if(n==="complex64"){const $=Ut(e);return[zt($[0],0,n)]}return n==="bool"?[os(e[0])]:[e[0].toString()]}if(c===1){if(a>bn){const w=qt*i;let v=Array.from(e.slice(0,w)),A=Array.from(e.slice((a-qt)*i,a*i));return n==="complex64"&&(v=Ut(v),A=Ut(A)),["["+v.map((M,T)=>zt(M,s[T],n)).join(", ")+", ..., "+A.map((M,T)=>zt(M,s[a-qt+T],n)).join(", ")+"]"]}return["["+(n==="complex64"?Ut(e):Array.from(e)).map((w,v)=>zt(w,s[v],n)).join(", ")+"]"]}const u=t.slice(1),h=r.slice(1),l=r[0]*i,p=[];if(a>bn){for(let $=0;$<qt;$++){const w=$*l,v=w+l;p.push(...re(e.slice(w,v),u,n,h,s,!1))}p.push("...");for(let $=a-qt;$<a;$++){const w=$*l,v=w+l;p.push(...re(e.slice(w,v),u,n,h,s,$===a-1))}}else for(let $=0;$<a;$++){const w=$*l,v=w+l;p.push(...re(e.slice(w,v),u,n,h,s,$===a-1))}const m=c===2?",":"";p[0]="["+(a>0?p[0]+m:"");for(let $=1;$<p.length-1;$++)p[$]=" "+p[$]+m;let k=`,
`;for(let $=2;$<c;$++)k+=`
`;return p[p.length-1]=" "+p[p.length-1]+"]"+(o?"":k),p}function Ut(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Aa{constructor(t,n,r){if(this.dtype=n,this.shape=t.slice(),this.size=J(t),r!=null){const s=r.length;f(s===this.size,()=>`Length of values '${s}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=r||js(n,this.size),this.strides=Oe(t)}set(t,...n){n.length===0&&(n=[0]),f(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const r=this.locToIndex(n);this.values[r]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const s of t){if(s<0||s>=this.shape[n]){const o=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(o)}n++}let r=t[t.length-1];for(let s=0;s<t.length-1;++s)r+=this.strides[s]*t[s];return this.values[r]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let r=0;r<t.length-1;++r)n+=this.strides[r]*t[r];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let r=0;r<n.length-1;++r)n[r]=Math.floor(t/this.strides[r]),t-=n[r]*this.strides[r];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return st().makeTensor(this.values,this.shape,this.dtype)}}let st=null,Mt=null;function Ca(e){st=e}function Fa(e){Mt=e}class tt{constructor(t,n,r,s){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=J(t),this.strides=Oe(t),this.dataId=r,this.id=s,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return Mt.buffer(this.shape,this.dtype,t)}bufferSync(){return Mt.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return un(this.shape,t,this.dtype==="complex64")}arraySync(){return un(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=st().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(r=>mn(r))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),st().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=st().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>mn(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await st().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(st().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return Mt.print(this,t)}clone(){return this.throwIfDisposed(),Mt.clone(this)}toString(t=!1){const n=this.dataSync();return Na(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),Mt.cast(this,t)}variable(t=!0,n,r){return this.throwIfDisposed(),st().makeVariable(this,t,n,r)}}Object.defineProperty(tt,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function Ba(){return Ue("Tensor",()=>tt)}Ba();class ce extends tt{constructor(t,n,r,s){super(t.shape,t.dtype,t.dataId,s),this.trainable=n,this.name=r}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!pe(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);st().disposeTensor(this),this.dataId=t.dataId,st().incRef(this,null)}dispose(){st().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(ce,Symbol.hasInstance,{value:e=>e instanceof tt&&e.assign!=null&&e.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var yn;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(yn||(yn={}));var Ae;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(Ae||(Ae={}));var Ce;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(Ce||(Ce={}));var Fe;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(Fe||(Fe={}));var Be;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(Be||(Be={}));const Ma={float32:Fe,int32:Ae,bool:Ce,complex64:Be};function _a(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return Ma[e][t]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function V(e,t){if(e.dtype===t.dtype)return[e,t];const n=_a(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function as(e){const t=[];return is(e,t,new Set),t}function is(e,t,n){if(e==null)return;if(e instanceof tt){t.push(e);return}if(!Pa(e))return;const r=e;for(const s in r){const o=r[s];n.has(o)||(n.add(o),is(o,t,n))}}function Pa(e){return Array.isArray(e)||typeof e=="object"}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $e(e){return e.kernelName!=null}class kn{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class Gt{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new kn}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const r=t[n];if(await this.initializeBackend(r).success){await this.setBackend(r);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,r=1){return t in this.registryFactory?(Bt(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:r},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:r}=this.initializeBackend(t);if(!(r?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new va(this.backendInstance),!0}setupRegisteredKernels(){gn(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){gn(t).forEach(r=>{r.disposeFunc!=null&&r.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const r=n.factory();if(r&&!(r instanceof Us)&&typeof r.then=="function"){const s=++this.pendingBackendInitId,o=r.then(i=>s<this.pendingBackendInitId?!1:(this.registry[t]=i,this.pendingBackendInit=null,!0)).catch(i=>(s<this.pendingBackendInitId||(this.pendingBackendInit=null,Bt(`Initialization of backend ${t} failed`),Bt(i.stack||i.message)),!1));return this.pendingBackendInit=o,{success:o,asyncInit:!0}}else return this.registry[t]=r,{success:!0,asyncInit:!1}}catch(r){return Bt(`Initialization of backend ${t} failed`),Bt(r.stack||r.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const r=t[n],{success:s,asyncInit:o}=this.initializeBackend(r);if(o||s)return{name:r,asyncInit:o}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const r=this.state.tensorInfo.get(n),s=r.backend,o=this.readSync(n),i=s.refCount(n);s.disposeData(n,!0),r.backend=t,t.move(n,o,r.shape,r.dtype,i),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let r=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");r=t}let s;return this.scopedRun(()=>this.startScope(r),()=>this.endScope(s),()=>(s=n(),s instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),s))}scopedRun(t,n,r){t();try{const s=r();return n(),s}catch(s){throw n(),s}}nextTensorId(){return Gt.nextTensorId++}nextVariableId(){return Gt.nextVariableId++}clone(t){const n=g.runKernel(He,{x:t}),r={x:t},s=i=>({x:()=>{const a="float32",c={x:i},u={dtype:a};return g.runKernel(Ve,c,u)}}),o=[];return this.addTapeNode(this.state.activeScope.name,r,[n],s,o,{}),n}runKernel(t,n,r){if(this.backendName==null&&this.backend,!(De(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:r})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,r){const s=this.backend.numDataIds();let o=0;r.forEach(c=>{o+=c.dtype==="complex64"?3:1});const i=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=s-n-o-i;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${t}'`)}runKernelFunc(t){let n,r=[];const s=this.isTapeOn(),o=this.state.numBytes,i=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let a;this.backendName==null&&this.backend;let c;const u=$e(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if($e(t)){const{kernelName:k,inputs:$,attrs:w}=t;this.backendName==null&&this.backend;const v=De(k,this.backendName);f(v!=null,()=>`Cannot find registered kernel '${k}' for backend '${this.backendName}'`),a=()=>{const A=this.backend.numDataIds();c=v.kernelFunc({inputs:$,attrs:w,backend:this.backend});const M=Array.isArray(c)?c:[c];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(k,A,M);const T=M.map(_=>_.rank!=null?_:this.makeTensorFromTensorInfo(_));if(s){const _=this.getTensorsForGradient(k,$,T);r=this.saveTensorsForBackwardMode(_)}return T}}else{const{forwardFunc:k}=t,$=w=>{s&&(r=w.map(v=>this.keep(this.clone(v))))};a=()=>{const w=this.backend.numDataIds();c=this.tidy(()=>k(this.backend,$));const v=Array.isArray(c)?c:[c];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(u,w,v),v}}const{inputs:h,attrs:l}=t,p=$e(t)?null:t.backwardsFunc;let m;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=a():(m=this.profiler.profileKernel(u,h,()=>a()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(m),n=m.outputs)}),s&&this.addTapeNode(u,h,n,p,r,l),this.state.profiling&&this.state.activeProfile.kernels.push({name:u,bytesAdded:this.state.numBytes-o,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-i,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(k=>h[k]!=null?h[k].shape:null),outputShapes:n.map(k=>k.shape),kernelTimeMs:m.timeMs,extraInfo:m.extraInfo}),Array.isArray(c)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(r=>this.keep(this.clone(r)))}getTensorsForGradient(t,n,r){const s=pn(t);if(s!=null){const o=s.inputsToSave||[],i=s.outputsToSave||[];let a;s.saveAllInputs?(f(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(n).map(u=>n[u])):a=o.map(u=>n[u]);const c=r.filter((u,h)=>i[h]);return a.concat(c)}return[]}makeTensor(t,n,r,s){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");r=r||"float32",s=s||this.backend;let o=t;r==="string"&&Le(t[0])&&(o=t.map(c=>$a(c)));const i=s.write(o,n,r),a=new tt(n,r,i,this.nextTensorId());if(this.trackTensor(a,s),r==="string"){const c=this.state.tensorInfo.get(i),u=Js(o);this.state.numBytes+=u-c.bytes,c.bytes=u}return a}makeTensorFromDataId(t,n,r,s){r=r||"float32";const o={dataId:t,shape:n,dtype:r};return this.makeTensorFromTensorInfo(o,s)}makeTensorFromTensorInfo(t,n){const{dataId:r,shape:s,dtype:o}=t,i=new tt(s,o,r,this.nextTensorId());return this.trackTensor(i,n),i}makeVariable(t,n=!0,r,s){r=r||this.nextVariableId().toString(),s!=null&&s!==t.dtype&&(t=t.cast(s));const o=new ce(t,n,r,this.nextTensorId());if(this.state.registeredVariables[o.name]!=null)throw new Error(`Variable with name ${o.name} was already registered`);return this.state.registeredVariables[o.name]=o,this.incRef(o,this.backend),o}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let r=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(r=t.size*Te(t.dtype)),this.state.numBytes+=r,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:r})),t instanceof ce||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const r=t.size*Te(t.dtype);this.state.numBytes-=r}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,r=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(s=>s.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-r;for(const s of this.state.activeProfile.kernels)s.kernelTimeMs=await s.kernelTimeMs,s.extraInfo=await s.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,r,s,o,i){const a={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:r,saved:o},c=pn(t);c!=null&&(s=c.gradFunc),s!=null&&(a.gradient=u=>(u=u.map((h,l)=>{if(h==null){const p=r[l],m=qe(p.size,p.dtype);return this.makeTensor(m,p.shape,p.dtype)}return h}),s(u.length>1?u:u[0],o,i))),this.state.activeTape.push(a)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=as(t),r=new Set(n.map(o=>o.id));for(let o=0;o<this.state.activeScope.track.length;o++){const i=this.state.activeScope.track[o];!i.kept&&!r.has(i.id)&&i.dispose()}const s=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(o=>{!o.kept&&o.scopeId===s.id&&this.track(o)})}gradients(t,n,r,s=!1){if(f(n.length>0,()=>"gradients() received an empty list of xs."),r!=null&&r.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${r.dtype}'`);const o=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));f(o instanceof tt,()=>"The result y returned by f() must be a tensor.");const i=Ta(this.state.activeTape,n,o);if(!s&&i.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const a={};a[o.id]=r??Ga(o.shape),Ia(a,i,u=>this.tidy(u),Ra);const c=n.map(u=>a[u.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(u=>{for(const h of u.saved)h.dispose()}),this.state.activeTape=null),{value:o,grads:c}})}customGrad(t){return f(Ie(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{f(n.every(a=>a instanceof tt),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let r;const s={};n.forEach((a,c)=>{s[c]=a});const o=(a,c)=>(r=t(...n,c),f(r.value instanceof tt,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),f(Ie(r.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),r.value),i=(a,c)=>{const u=r.gradFunc(a,c),h=Array.isArray(u)?u:[u];f(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),f(h.every(p=>p instanceof tt),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const l={};return h.forEach((p,m)=>{l[m]=()=>p}),l};return this.runKernelFunc({forwardFunc:o,backwardsFunc:i,inputs:s})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=ie(),r=await this.backend.time(t);return r.wallMs=ie()-n,r}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new kn;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}Gt.nextTensorId=0;Gt.nextVariableId=0;function Ga(e){const t=Ln(J(e),"float32");return g.makeTensor(t,e,"float32")}function cs(){const e=On();if(e._tfengine==null){const t=new to(e);e._tfengine=new Gt(t)}return so(e._tfengine.ENV),Ca(()=>e._tfengine),e._tfengine}const g=cs();function Ra(e,t){const n={a:e,b:t};return g.runKernel(We,n)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function La(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const et=F();et.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});et.registerFlag("IS_BROWSER",()=>La());et.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");et.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));et.registerFlag("PROD",()=>!1);et.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>et.getBool("DEBUG"));et.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);et.registerFlag("IS_TEST",()=>!1);et.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>!0);et.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);et.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);et.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jt(e,t){let n=e;if(ot(e))return t==="string"?[]:[e.length];if(typeof e=="object"){if("texture"in e){const o=e.channels||"RGBA";return[e.height,e.width*o.length]}else if("buffer"in e&&!(e.buffer instanceof ArrayBuffer))return[e.buffer.size/(t==null?4:Te(t))]}if(!Array.isArray(e))return[];const s=[];for(;Array.isArray(n)||ot(n)&&t!=="string";)s.push(n.length),n=n[0];return Array.isArray(e)&&F().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&us(e,s,[]),s}function us(e,t,n){if(n=n||[],!Array.isArray(e)&&!ot(e)){f(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}f(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),f(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const r=t.slice(1);for(let s=0;s<e.length;++s)us(e[s],r,n.concat(s))}function wn(e,t,n,r){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function d(e,t,n,r="numeric"){if(e instanceof tt)return wn(r,e.dtype,t,n),e;let s=Ke(e);if(s!=="string"&&["bool","int32","float32"].indexOf(r)>=0&&(s=r),wn(r,s,t,n),e==null||!ot(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const c=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${c}'`)}const o=Jt(e,s);!ot(e)&&!Array.isArray(e)&&(e=[e]);const a=s!=="string"?ss(e,s):Vt(e,[],!0);return g.makeTensor(a,o,s)}function ls(e,t,n,r="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((o,i)=>d(o,`${t}[${i}]`,n,r))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ka="__op";function b(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const r=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+Ka;const s=(...o)=>{g.startScope(n);try{const i=r(...o);return ze(i)&&console.error("Cannot return a Promise inside of tidy."),g.endScope(i),i}catch(i){throw g.endScope(null),i}};return Object.defineProperty(s,"name",{value:n,configurable:!0}),s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oa(e,t){const n=d(e,"real","complex"),r=d(t,"imag","complex");Vs(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);const s={real:n,imag:r};return g.runKernel(vo,s)}const ge=b({complex_:Oa});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zt(e,t,n,r){if(r==null)r=Ke(e);else if(r==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(typeof e=="object"&&("texture"in e||"buffer"in e&&!(e.buffer instanceof ArrayBuffer))){if(r!=="float32"&&r!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return g.backend.createTensorFromGPUData(e,t||n,r)}if(!ot(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){bt(t);const s=J(t),o=J(n);f(s===o,()=>`Based on the provided shape, [${t}], the tensor should have ${s} values but has ${o}`);for(let i=0;i<n.length;++i){const a=n[i],c=i===n.length-1?a!==J(t.slice(i)):!0;f(n[i]===t[i]||!c,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!ot(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=r!=="string"?ss(e,r):Vt(e,[],!0),g.makeTensor(e,t,r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function se(e,t,n){const r=Jt(e,n);return Zt(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xn={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ue=4;async function Ad(e,t){const n=[],r=[],s=Array.isArray(e)?e.map(i=>i.name):Object.keys(e);for(let i=0;i<s.length;++i){const a=s[i],c=Array.isArray(e)?e[i].tensor:e[a];if(c.dtype!=="float32"&&c.dtype!=="int32"&&c.dtype!=="bool"&&c.dtype!=="string"&&c.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${a}': ${c.dtype}`);const u={name:a,shape:c.shape,dtype:c.dtype};if(c.dtype==="string"){const h=new Promise(async l=>{const p=await c.bytes(),m=p.reduce((w,v)=>w+v.length,0)+ue*p.length,k=new Uint8Array(m);let $=0;for(let w=0;w<p.length;w++){const v=p[w],A=new Uint8Array(new Uint32Array([v.length]).buffer);k.set(A,$),$+=ue,k.set(v,$),$+=v.length}l(k)});r.push(h)}else r.push(c.data());t!=null&&(u.group=t),n.push(u)}const o=await Promise.all(r);return{data:qa(o),specs:n}}function Cd(e,t){const n={};let r,s=0;for(const o of t){const i=o.name,a=o.dtype,c=o.shape,u=J(c);let h;if("quantization"in o){const l=o.quantization;if(l.dtype==="uint8"||l.dtype==="uint16"){if(!("min"in l&&"scale"in l))throw new Error(`Weight ${o.name} with quantization ${l.dtype} doesn't have corresponding metadata min and scale.`)}else if(l.dtype==="float16"){if(a!=="float32")throw new Error(`Weight ${o.name} is quantized with ${l.dtype} which only supports weights of type float32 not ${a}.`)}else throw new Error(`Weight ${o.name} has unknown quantization dtype ${l.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const p=xn[l.dtype],m=e.slice(s,s+u*p),k=l.dtype==="uint8"?new Uint8Array(m):new Uint16Array(m);if(a==="float32")if(l.dtype==="uint8"||l.dtype==="uint16"){h=new Float32Array(k.length);for(let $=0;$<k.length;$++){const w=k[$];h[$]=w*l.scale+l.min}}else if(l.dtype==="float16")r===void 0&&(r=Qa()),h=r(k);else throw new Error(`Unsupported quantization type ${l.dtype} for weight type float32.`);else if(a==="int32"){if(l.dtype!=="uint8"&&l.dtype!=="uint16")throw new Error(`Unsupported quantization type ${l.dtype} for weight type int32.`);h=new Int32Array(k.length);for(let $=0;$<k.length;$++){const w=k[$];h[$]=Math.round(w*l.scale+l.min)}}else throw new Error(`Unsupported dtype in weight '${i}': ${a}`);s+=u*p}else if(a==="string"){const l=J(o.shape);h=[];for(let p=0;p<l;p++){const m=new Uint32Array(e.slice(s,s+ue))[0];s+=ue;const k=new Uint8Array(e.slice(s,s+m));h.push(k),s+=m}}else{const l=xn[a],p=e.slice(s,s+u*l);if(a==="float32")h=new Float32Array(p);else if(a==="int32")h=new Int32Array(p);else if(a==="bool")h=new Uint8Array(p);else if(a==="complex64"){h=new Float32Array(p);const m=new Float32Array(h.length/2),k=new Float32Array(h.length/2);for(let v=0;v<m.length;v++)m[v]=h[v*2],k[v]=h[v*2+1];const $=se(m,c,"float32"),w=se(k,c,"float32");n[i]=ge($,w),$.dispose(),w.dispose()}else throw new Error(`Unsupported dtype in weight '${i}': ${a}`);s+=u*l}a!=="complex64"&&(n[i]=se(h,c,a))}return n}function qa(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(o=>{if(t+=o.byteLength,n.push(o.byteLength===o.buffer.byteLength?o:new o.constructor(o)),!(o instanceof Float32Array||o instanceof Int32Array||o instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${o.constructor.name}`)});const r=new Uint8Array(t);let s=0;return n.forEach(o=>{r.set(new Uint8Array(o.buffer),s),s+=o.byteLength}),r.buffer}const Xe=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function $n(e){return Xe?Buffer.byteLength(e):new Blob([e]).size}function za(e){if(Xe)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let r=0,s=t.length;r<s;r++)n+=String.fromCharCode(t[r]);return btoa(n)}function Ua(e){if(Xe){const r=Buffer.from(e,"base64");return r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let r=0;r<t.length;++r)n.set([t.charCodeAt(r)],r);return n.buffer}function Wa(e){if(e.length===1)return e[0];let t=0;e.forEach(s=>{t+=s.byteLength});const n=new Uint8Array(t);let r=0;return e.forEach(s=>{n.set(new Uint8Array(s),r),r+=s.byteLength}),n.buffer}function Va(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function Ha(e,t,n){const r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(r.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");r.weightSpecs=t,r.weightData=n}return e.signature!=null&&(r.signature=e.signature),e.userDefinedMetadata!=null&&(r.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(r.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(r.initializerSignature=e.initializerSignature),r}async function ja(e,t){let n,r;return e.weightsManifest!=null&&([n,r]=await t(e.weightsManifest)),Ha(e,n,r)}function Ye(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:$n(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:$n(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:e.weightData.byteLength}}function Xa(e){const t=[];for(const n of e)t.push(...n.weights);return t}function Ya(){const e=n=>{let r=n<<13,s=0;for(;!(r&8388608);)s-=8388608,r<<=1;return r&=-8388609,s+=947912704,r|s},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function Ja(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function Za(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function Qa(){const e=Ya(),t=Ja(),n=Za();return r=>{const s=new ArrayBuffer(4*r.length),o=new Uint32Array(s);for(let i=0;i<r.length;i++){const a=r[i],c=e[n[a>>10]+(a&1023)]+t[a>>10];o[i]=c}return new Float32Array(s)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class z{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return z.instance==null&&(z.instance=new z),z.instance}static registerSaveRouter(t){z.getInstance().saveRouters.push(t)}static registerLoadRouter(t){z.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return z.getHandlers(t,"save")}static getLoadHandlers(t,n){return z.getHandlers(t,"load",n)}static getHandlers(t,n,r){const s=[];return(n==="load"?z.getInstance().loadRouters:z.getInstance().saveRouters).forEach(i=>{const a=i(t,r);a!==null&&s.push(a)}),s}}const Fd=e=>z.getSaveHandlers(e),Bd=(e,t)=>z.getLoadHandlers(e,t);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Me="tensorflowjs",_e=1,Et="models_store",yt="model_info_store";function hs(){if(!F().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function Pe(e){const t=e.result;t.createObjectStore(Et,{keyPath:"modelPath"}),t.createObjectStore(yt,{keyPath:"modelPath"})}class Tt{constructor(t){if(this.indexedDB=hs(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((r,s)=>{const o=this.indexedDB.open(Me,_e);o.onupgradeneeded=()=>Pe(o),o.onsuccess=()=>{const i=o.result;if(n==null){const a=i.transaction(Et,"readonly"),u=a.objectStore(Et).get(this.modelPath);u.onsuccess=()=>{if(u.result==null)return i.close(),s(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));r(u.result.modelArtifacts)},u.onerror=h=>(i.close(),s(u.error)),a.oncomplete=()=>i.close()}else{const a=Ye(n),c=i.transaction(yt,"readwrite");let u=c.objectStore(yt);const h=u.put({modelPath:this.modelPath,modelArtifactsInfo:a});let l;h.onsuccess=()=>{l=i.transaction(Et,"readwrite");const m=l.objectStore(Et).put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:a});m.onsuccess=()=>r({modelArtifactsInfo:a}),m.onerror=k=>{u=c.objectStore(yt);const $=u.delete(this.modelPath);$.onsuccess=()=>(i.close(),s(m.error)),$.onerror=w=>(i.close(),s(m.error))}},h.onerror=p=>(i.close(),s(h.error)),c.oncomplete=()=>{l==null?i.close():l.oncomplete=()=>i.close()}}},o.onerror=i=>s(o.error)})}}Tt.URL_SCHEME="indexeddb://";const fs=e=>F().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Tt.URL_SCHEME)?ti(e.slice(Tt.URL_SCHEME.length)):null;z.registerSaveRouter(fs);z.registerLoadRouter(fs);function ti(e){return new Tt(e)}function ei(e){return e.startsWith(Tt.URL_SCHEME)?e.slice(Tt.URL_SCHEME.length):e}class ni{constructor(){this.indexedDB=hs()}async listModels(){return new Promise((t,n)=>{const r=this.indexedDB.open(Me,_e);r.onupgradeneeded=()=>Pe(r),r.onsuccess=()=>{const s=r.result,o=s.transaction(yt,"readonly"),a=o.objectStore(yt).getAll();a.onsuccess=()=>{const c={};for(const u of a.result)c[u.modelPath]=u.modelArtifactsInfo;t(c)},a.onerror=c=>(s.close(),n(a.error)),o.oncomplete=()=>s.close()},r.onerror=s=>n(r.error)})}async removeModel(t){return t=ei(t),new Promise((n,r)=>{const s=this.indexedDB.open(Me,_e);s.onupgradeneeded=()=>Pe(s),s.onsuccess=()=>{const o=s.result,i=o.transaction(yt,"readwrite"),a=i.objectStore(yt),c=a.get(t);let u;c.onsuccess=()=>{if(c.result==null)return o.close(),r(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const h=a.delete(t),l=()=>{u=o.transaction(Et,"readwrite");const m=u.objectStore(Et).delete(t);m.onsuccess=()=>n(c.result.modelArtifactsInfo),m.onerror=k=>r(c.error)};h.onsuccess=l,h.onerror=p=>(l(),o.close(),r(c.error))}},c.onerror=h=>(o.close(),r(c.error)),i.oncomplete=()=>{u==null?o.close():u.oncomplete=()=>o.close()}},s.onerror=o=>r(s.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gt="/",_t="tensorflowjs_models",ds="info",ri="model_topology",si="weight_specs",oi="weight_data",ai="model_metadata";function ps(e){return{info:[_t,e,ds].join(gt),topology:[_t,e,ri].join(gt),weightSpecs:[_t,e,si].join(gt),weightData:[_t,e,oi].join(gt),modelMetadata:[_t,e,ai].join(gt)}}function gs(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function ii(e){const t=e.split(gt);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(gt)}function ci(e){return e.startsWith(It.URL_SCHEME)?e.slice(It.URL_SCHEME.length):e}class It{constructor(t){if(!F().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=ps(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),r=JSON.stringify(t.weightSpecs),s=Ye(t);try{this.LS.setItem(this.keys.info,JSON.stringify(s)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,r),this.LS.setItem(this.keys.weightData,za(t.weightData));const o={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(o)),{modelArtifactsInfo:s}}catch{throw gs(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${s.modelTopologyBytes}, weightSpecsBytes=${s.weightSpecsBytes}, weightDataBytes=${s.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},r=JSON.parse(this.LS.getItem(this.keys.topology));if(r==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=r;const s=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(s==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=s;const o=this.LS.getItem(this.keys.modelMetadata);if(o!=null){const a=JSON.parse(o);n.format=a.format,n.generatedBy=a.generatedBy,n.convertedBy=a.convertedBy,a.signature!=null&&(n.signature=a.signature),a.userDefinedMetadata!=null&&(n.userDefinedMetadata=a.userDefinedMetadata),a.modelInitializer!=null&&(n.modelInitializer=a.modelInitializer),a.initializerSignature!=null&&(n.initializerSignature=a.initializerSignature),a.trainingConfig!=null&&(n.trainingConfig=a.trainingConfig)}const i=this.LS.getItem(this.keys.weightData);if(i==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=Ua(i),n}}It.URL_SCHEME="localstorage://";const ms=e=>F().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(It.URL_SCHEME)?ui(e.slice(It.URL_SCHEME.length)):null;z.registerSaveRouter(ms);z.registerLoadRouter(ms);function ui(e){return new It(e)}class li{constructor(){f(F().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),f(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=_t+gt,r=gt+ds;for(let s=0;s<this.LS.length;++s){const o=this.LS.key(s);if(o.startsWith(n)&&o.endsWith(r)){const i=ii(o);t[i]=JSON.parse(this.LS.getItem(o))}}return t}async removeModel(t){t=ci(t);const n=ps(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const r=JSON.parse(this.LS.getItem(n.info));return gs(n),r}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vn="://";class lt{constructor(){this.managers={}}static getInstance(){return lt.instance==null&&(lt.instance=new lt),lt.instance}static registerManager(t,n){f(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(vn)&&(t=t.slice(0,t.indexOf(vn))),f(t.length>0,()=>"scheme must not be an empty string.");const r=lt.getInstance();f(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){const n=lt.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(lt.getInstance().managers)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hi{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(typeof window>"u"||!F().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",r=>{if(r.source===window&&r.data.name===this.messageName){r.stopPropagation();const s=this.functionRefs[r.data.index];s(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return t instanceof Float32Array||t instanceof Int32Array||t instanceof Uint8Array||t instanceof Uint8ClampedArray}}if(F().get("IS_BROWSER")){F().setPlatform("browser",new hi);try{lt.registerManager(It.URL_SCHEME,new li)}catch{}try{lt.registerManager(Tt.URL_SCHEME,new ni)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fi={importFetch:()=>require("node-fetch")};let ve;class di{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return F().global.fetch!=null?F().global.fetch(t,n):(ve==null&&(ve=fi.importFetch()),ve(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}F().get("IS_NODE")&&!F().get("IS_BROWSER")&&F().setPlatform("node",new di);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qt(e,t="float32",n){return t=t||"float32",bt(e),new Aa(e,t,n)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pi(e,t){const n=d(e,"x","cast");if(!Ys(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const r={x:n},s={dtype:t};return g.runKernel(Ve,r,s)}const S=b({cast_:pi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gi(e){const n={x:d(e,"x","clone","string_or_numeric")};return g.runKernel(He,n)}const Pt=b({clone_:gi});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mi(e,t=!1){console.log(e.toString(t))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */cs();const bi={buffer:Qt,cast:S,clone:Pt,print:mi};Fa(bi);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Md(){return g.memory()}function q(e,t){return g.tidy(e,t)}function Z(e){as(e).forEach(n=>n.dispose())}function yi(e){return g.keep(e)}function _d(){return g.backend}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ki(e,t){let n=d(e,"a","add"),r=d(t,"b","add");[n,r]=V(n,r);const s={a:n,b:r};return g.runKernel(We,s)}const E=b({add_:ki});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wi(e,t){let n=d(e,"a","floorDiv"),r=d(t,"b","floorDiv");[n,r]=V(n,r);const s={a:n,b:r};return g.runKernel(ur,s)}const xi=b({floorDiv_:wi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $i(e,t){let n=d(e,"a","div"),r=d(t,"b","div");if([n,r]=V(n,r),n.dtype==="int32"&&r.dtype==="int32")return xi(n,r);const s={a:n,b:r},o={};return g.runKernel(sr,s,o)}const N=b({div_:$i});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vi(e,t){let n=d(e,"a","mul"),r=d(t,"b","mul");[n,r]=V(n,r);const s={a:n,b:r};return g.runKernel(vr,s)}const y=b({mul_:vi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Si(e){const t=d(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return g.runKernel(Xn,n)}else{const n={x:t};return g.runKernel(qn,n)}}const pt=b({abs_:Si});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ei(e,t=null,n=!1){const s={x:d(e,"x","all","bool")},o={axis:t,keepDims:n};return g.runKernel(uo,s,o)}const Pd=b({all_:Ei});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ti(e,t=null,n=!1){const s={x:d(e,"x","any","bool")},o={axis:t,keepDims:n};return g.runKernel(lo,s,o)}const Gd=b({any_:Ti});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ii(e,t=0){const r={x:d(e,"x","argMax")},s={axis:t};return g.runKernel(zn,r,s)}const Rd=b({argMax_:Ii});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ni(e,t,n,r,s,o,i=!1,a="channelsLast"){let[c,u,h,l]=[-1,-1,-1,-1];if(a==="channelsLast")[c,u,h,l]=e;else if(a==="channelsFirst")[c,l,u,h]=e;else throw new Error(`Unknown dataFormat ${a}`);const[p,m,,k]=t,[$,w]=le(n),[v,A]=le(r),M=Ge(p,v),T=Ge(m,A),{padInfo:_,outHeight:D,outWidth:j}=Ci(s,u,h,$,w,M,T,o,a),O=i?k*l:k;let H;return a==="channelsFirst"?H=[c,O,D,j]:a==="channelsLast"&&(H=[c,D,j,O]),{batchSize:c,dataFormat:a,inHeight:u,inWidth:h,inChannels:l,outHeight:D,outWidth:j,outChannels:O,padInfo:_,strideHeight:$,strideWidth:w,filterHeight:p,filterWidth:m,effectiveFilterHeight:M,effectiveFilterWidth:T,dilationHeight:v,dilationWidth:A,inShape:e,outShape:H,filterShape:t}}function Di(e,t,n,r,s){r==null&&(r=Ai(e,t,n));const o=e[0],i=e[1],a=he((o-t+2*r)/n+1,s),c=he((i-t+2*r)/n+1,s);return[a,c]}function Ai(e,t,n,r=1){const s=Ge(t,r);return Math.floor((e[0]*(n-1)-n+s)/2)}function le(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function Ge(e,t){return t<=1?e:e+(e-1)*(t-1)}function Ci(e,t,n,r,s,o,i,a,c){let u,h,l;if(typeof e=="number"){u={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const m=Di([t,n],o,r,e,a);h=m[0],l=m[1]}else if(e==="same"){h=Math.ceil(t/r),l=Math.ceil(n/s);const p=Math.max(0,(h-1)*r+o-t),m=Math.max(0,(l-1)*s+i-n),k=Math.floor(p/2),$=p-k,w=Math.floor(m/2),v=m-w;u={top:k,bottom:$,left:w,right:v,type:"SAME"}}else if(e==="valid")u={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((t-o+1)/r),l=Math.ceil((n-i+1)/s);else if(typeof e=="object"){const p=c==="channelsLast"?e[1][0]:e[2][0],m=c==="channelsLast"?e[1][1]:e[2][1],k=c==="channelsLast"?e[2][0]:e[3][0],$=c==="channelsLast"?e[2][1]:e[3][1];u={top:p,bottom:m,left:k,right:$,type:p===0&&m===0&&k===0&&$===0?"VALID":"EXPLICIT"},h=he((t-o+p+m)/r+1,a),l=he((n-i+k+$)/s+1,a)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:u,outHeight:h,outWidth:l}}function he(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function Rt(e){const[t,n,r]=le(e);return t===1&&n===1&&r===1}function Nt(e,t){return Rt(e)||Rt(t)}function Lt(e){return le(e).every(t=>t>0)}function Q(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")f(ae(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(r=>{r.forEach(s=>{f(ae(s),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${s}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fi(e,t){const r={x:d(e,"x","reshape","string_or_numeric")},s={shape:t};return g.runKernel(Fr,r,s)}const x=b({reshape_:Fi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bi(e,t,n,r,s){const o=d(e,"x","avgPool","float32"),i=1;f(Nt(n,i),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${i}'`);let a=o,c=!1;o.rank===3&&(c=!0,a=x(o,[1,o.shape[0],o.shape[1],o.shape[2]])),f(a.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${a.rank}.`),Q("avgPool",r,s);const u={x:a},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s};let l=g.runKernel(Un,u,h);return l=S(l,o.dtype),c?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const Ld=b({avgPool_:Bi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mi(e,t,n,r,s,o="NDHWC"){const i=d(e,"x","avgPool3d","float32");let a=i,c=!1;i.rank===4&&(c=!0,a=x(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),f(a.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${a.rank}.`),f(o==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${o}`),f(typeof n=="number"&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),Q("avgPool3d",r,s);const u={x:a},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s,dataFormat:o};let l=g.runKernel(Wn,u,h);return l=S(l,a.dtype),c?x(l,[l.shape[1],l.shape[2],l.shape[3],l.shape[4]]):l}const Kd=b({avgPool3d_:Mi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _i(e,t=0){f(e.length>=1,()=>"Pass at least one tensor to concat");const n=ls(e,"tensors","concat","string_or_numeric");if(n[0].dtype==="complex64"&&n.forEach(o=>{if(o.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${o.dtype}. `)}),n.length===1)return Pt(n[0]);const r=n,s={axis:t};return g.runKernel(Yn,r,s)}const kt=b({concat_:_i});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pi(e,t,n=!1,r=!1){let s=d(e,"a","matMul"),o=d(t,"b","matMul");[s,o]=V(s,o);const i={a:s,b:o},a={transposeA:n,transposeB:r};return g.runKernel(Vn,i,a)}const G=b({matMul_:Pi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gi(e){const n={x:d(e,"x","sigmoid","float32")};return g.runKernel(Ur,n)}const bs=b({sigmoid_:Gi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ri(e,t,n){const r=d(e,"x","slice","string_or_numeric");if(r.rank===0)throw new Error("Slicing scalar is not possible");const s={x:r},o={begin:t,size:n};return g.runKernel(Or,s,o)}const K=b({slice_:Ri});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Li(e){const n={x:d(e,"x","tanh","float32")};return g.runKernel(Zr,n)}const Od=b({tanh_:Li});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ki(e,t,n){const r=d(e,"x","batchToSpaceND"),s=t.reduce((a,c)=>a*c);f(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),f(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),f(r.shape[0]%s===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${s}`);const o={x:r},i={blockShape:t,crops:n};return g.runKernel(Hn,o,i)}const Oi=b({batchToSpaceND_:Ki});function qi(e){let t;return e.rank===0||e.rank===1?t=x(e,[1,1,1,e.size]):e.rank===2?t=x(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?t=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]):t=e,t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zi(e,t,n,r,s,o){o==null&&(o=.001);const i=d(e,"x","batchNorm"),a=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let u;s!=null&&(u=d(s,"scale","batchNorm"));let h;r!=null&&(h=d(r,"offset","batchNorm")),f(a.rank===c.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),f(h==null||a.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),f(u==null||a.rank===u.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const p={x:qi(i),scale:u,offset:h,mean:a,variance:c},m={varianceEpsilon:o},k=g.runKernel(lr,p,m);return x(k,i.shape)}const Je=b({batchNorm_:zi});function Ui(e,t,n,r,s,o){const i=d(e,"x","batchNorm"),a=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let u;s!=null&&(u=d(s,"scale","batchNorm"));let h;return r!=null&&(h=d(r,"offset","batchNorm")),f(i.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${i.rank}.`),f(a.rank===2||a.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${a.rank}.`),f(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${c.rank}.`),u!=null&&f(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${u.rank}.`),h!=null&&f(h.rank===2||h.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${h.rank}.`),Je(i,a,c,h,u,o)}const qd=b({batchNorm2d_:Ui});function Wi(e,t,n,r,s,o){const i=d(e,"x","batchNorm"),a=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let u;s!=null&&(u=d(s,"scale","batchNorm"));let h;return r!=null&&(h=d(r,"offset","batchNorm")),f(i.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${i.rank}.`),f(a.rank===3||a.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${a.rank}.`),f(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${c.rank}.`),u!=null&&f(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${u.rank}.`),h!=null&&f(h.rank===3||h.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${h.rank}.`),Je(i,a,c,h,u,o)}const zd=b({batchNorm3d_:Wi});function Vi(e,t,n,r,s,o){const i=d(e,"x","batchNorm"),a=d(t,"mean","batchNorm"),c=d(n,"variance","batchNorm");let u;s!=null&&(u=d(s,"scale","batchNorm"));let h;return r!=null&&(h=d(r,"offset","batchNorm")),f(i.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${i.rank}.`),f(a.rank===4||a.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${a.rank}.`),f(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${c.rank}.`),u!=null&&f(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${u.rank}.`),h!=null&&f(h.rank===4||h.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${h.rank}.`),Je(i,a,c,h,u,o)}const Ud=b({batchNorm4d_:Vi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hi(e,t,n){const r=d(e,"x","bincount"),s=d(t,"weights","bincount");f(r.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),f(n>=0,()=>`size must be non-negative, but got ${n}.`),f(s.size===r.size||s.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${s.shape}.`);const o={x:r,weights:s},i={size:n};return g.runKernel(wo,o,i)}const ji=b({bincount_:Hi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xi(e,t){let n=d(e,"broadcastTo","x");const r=n.shape;if(bt(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const u=n.shape.slice();for(;u.length<t.length;)u.unshift(1);n=x(n,u)}const s=n.shape,o=Array.from(t);for(let u=t.length-1;u>=0;u--)if(s[u]===t[u])o[u]=1;else if(n.shape[u]!==1)throw new Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(o.map((u,h)=>u>1?h:-1).filter(u=>u>=0).length===0)return Pt(n);const a={x:n},c={reps:o};return g.runKernel(je,a,c)}const Se=b({broadcastTo_:Xi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ze(e,t,n){bt(e);const r={shape:e,value:t,dtype:n};return g.runKernel(Lo,{},r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yi(e,t,n){const r=d(e,"x","clipByValue");if(f(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return Ze(r.shape,t,r.dtype);const s={x:r},o={clipValueMin:t,clipValueMax:n};return g.runKernel(jn,s,o)}const Wd=b({clipByValue_:Yi});function Ji(e){return kt(e,0)}const Vd=b({concat1d_:Ji});function Zi(e,t){return kt(e,t)}const Hd=b({concat2d_:Zi});function Qi(e,t){return kt(e,t)}const jd=b({concat3d_:Qi});function tc(e,t){return kt(e,t)}const Xd=b({concat4d_:tc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ec(e,t,n,r,s="NHWC",o=[1,1],i){const a=d(e,"x","conv2d","float32"),c=d(t,"filter","conv2d","float32");let u=a,h=!1;a.rank===3&&(h=!0,u=x(a,[1,a.shape[0],a.shape[1],a.shape[2]])),f(u.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${u.rank}.`),f(c.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${c.rank}.`),Q("conv2d",r,i);const l=s==="NHWC"?u.shape[3]:u.shape[1];f(l===c.shape[2],()=>`Error in conv2d: depth of input (${l}) must match input depth for filter ${c.shape[2]}.`),f(Nt(n,o),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),f(Lt(o),()=>"Error in conv2D: Dilated rates should be larger than 0."),f(Lt(n),()=>"Error in conv2D: Strides should be larger than 0.");const p={x:u,filter:c},m={strides:n,pad:r,dataFormat:s,dilations:o,dimRoundingMode:i},k=g.runKernel(Jn,p,m);return h?x(k,[k.shape[1],k.shape[2],k.shape[3]]):k}const me=b({conv2d_:ec});function nc(e,t,n,r,s="NWC",o=1,i){const a=d(e,"x","conv1d"),c=d(t,"filter","conv1d");let u=a,h=!1;a.rank===2&&(h=!0,u=x(a,[1,a.shape[0],a.shape[1]])),f(u.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${u.rank}.`),f(c.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${c.rank}.`),Q("conv1d",r,i),f(u.shape[2]===c.shape[1],()=>`Error in conv1d: depth of input (${u.shape[2]}) must match input depth for filter ${c.shape[1]}.`),f(Nt(n,o),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${o}'`),f(Lt(o),()=>"Error in conv1D: Dilated rates should be larger than 0."),f(Lt(n),()=>"Error in conv1D: Stride should be larger than 0."),f(s==="NWC",()=>`Error in conv1d: got dataFormat of ${s} but only NWC is currently supported.`);const l=x(c,[1,c.shape[0],c.shape[1],c.shape[2]]),p=x(u,[u.shape[0],1,u.shape[1],u.shape[2]]),w=me(p,l,[1,n],r,"NHWC",[1,o],i);return h?x(w,[w.shape[2],w.shape[3]]):x(w,[w.shape[0],w.shape[2],w.shape[3]])}const Yd=b({conv1d_:nc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rc(e,t,n,r,s,o="NHWC",i){f(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,c=t,u=!1;t.rank===3&&(u=!0,c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]),a=[1,e[0],e[1],e[2]]),f(a.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${a.length}.`),f(c.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${c.rank}`),f(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const h=o==="NHWC"?a[3]:a[1],l=o==="NHWC"?c.shape[3]:c.shape[1];f(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),f(l===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[3]}.`),Q("conv2dDerInput",s,i);const p={dy:c,filter:n},m={strides:r,pad:s,dataFormat:o,dimRoundingMode:i,inputShape:a},k=g.runKernel(Zn,p,m);return u?x(k,[k.shape[1],k.shape[2],k.shape[3]]):k}const Qe=b({conv2DBackpropInput_:rc});function sc(e,t,n,r,s,o){const i=d(e,"x","conv2dTranspose"),a=d(t,"filter","conv2dTranspose");return Qe(n,i,a,r,s,"NHWC",o)}const Jd=b({conv2dTranspose_:sc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oc(e,t,n,r,s="NDHWC",o=[1,1,1]){const i=d(e,"x","conv3d"),a=d(t,"filter","conv3d");let c=i,u=!1;i.rank===4&&(u=!0,c=x(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),f(c.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${c.rank}.`),f(a.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${a.rank}.`),f(c.shape[4]===a.shape[3],()=>`Error in conv3d: depth of input (${c.shape[4]}) must match input depth for filter ${a.shape[3]}.`),f(Nt(n,o),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),f(s==="NDHWC",()=>`Error in conv3d: got dataFormat of ${s} but only NDHWC is currently supported.`),f(Lt(o),()=>"Error in conv3D: Dilated rates should be larger than 0."),f(Lt(n),()=>"Error in conv3D: Strides should be larger than 0.");const h={x:c,filter:a},l={strides:n,pad:r,dataFormat:s,dilations:o},p=g.runKernel(Qn,h,l);return u?x(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const Zd=b({conv3d_:oc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ac(e,t,n,r,s){f(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let o=e,i=t,a=!1;t.rank===4&&(a=!0,i=x(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),o=[1,e[0],e[1],e[2],e[3]]);const c=o[4],u=i.shape[4];f(o.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${o.length}.`),f(i.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${i.rank}`),f(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),f(c===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[3]}.`),f(u===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${u}) must match output depth for filter ${n.shape[4]}.`);const h={dy:i,filter:n},l={pad:s,strides:r,inputShape:o},p=g.runKernel(To,h,l);return a?x(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const ys=b({conv3DBackpropInput_:ac});function ic(e,t,n,r,s){const o=d(e,"x","conv3dTranspose"),i=d(t,"filter","conv3dTranspose");return ys(n,o,i,r,s)}const Qd=b({conv3dTranspose_:ic});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cc(e){const n={x:d(e,"x","cos","float32")};return g.runKernel(tr,n)}const ks=b({cos_:cc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uc(e){const n={x:d(e,"x","cosh","float32")};return g.runKernel(er,n)}const lc=b({cosh_:uc});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hc(e,t=0,n=!1,r=!1){const o={x:d(e,"x","cumprod")},i={axis:t,exclusive:n,reverse:r};return g.runKernel(Io,o,i)}const Sn=b({cumprod_:hc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fc(e,t=0,n=!1,r=!1){const o={x:d(e,"x","cumsum")},i={axis:t,exclusive:n,reverse:r};return g.runKernel(nr,o,i)}const dc=b({cumsum_:fc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pc(e,t,n,r=!1){const s=d(e,"x","denseBincount"),o=d(t,"weights","denseBincount");f(s.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${s.dtype}`),f(s.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${s.rank}.`),f(n>=0,()=>`size must be non-negative, but got ${n}.`),f(o.size===s.size||o.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${s.shape}, weights shape: ${o.shape}.`);const i={x:s,weights:o},a={size:n,binaryOutput:r};return g.runKernel(Do,i,a)}const tp=b({denseBincount_:pc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gc(e,t,n,r,s="NHWC",o=[1,1],i){const a=d(e,"x","depthwiseConv2d","float32"),c=d(t,"filter","depthwiseConv2d","float32");let u=a,h=!1;a.rank===3&&(h=!0,u=x(a,[1,a.shape[0],a.shape[1],a.shape[2]])),f(u.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${u.rank}.`),f(c.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${c.rank}.`);const l=s==="NHWC"?u.shape[3]:u.shape[1];f(l===c.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${l}) must match the inChannels dimension in filter ${c.shape[2]}.`),Q("depthwiseConv2d",r,i);const p={x:u,filter:c},m={strides:n,pad:r,dataFormat:s,dilations:o,dimRoundingMode:i},k=g.runKernel(rr,p,m);return h?x(k,[k.shape[1],k.shape[2],k.shape[3]]):k}const mc=b({depthwiseConv2d_:gc});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U(e,t){const n=[];for(let r=0;r<t.length;r++){const s=e[e.length-r-1],o=t.length-r-1,i=t[o];(s==null||s===1&&i>1)&&n.unshift(o)}return n}function L(e,t){const n=[],r=Math.max(e.length,t.length);for(let s=0;s<r;s++){let o=e[e.length-s-1];o==null&&(o=1);let i=t[t.length-s-1];if(i==null&&(i=1),o===1)n.unshift(i);else if(i===1)n.unshift(o);else if(o!==i){const a=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(a)}else n.unshift(o)}return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bc(e,t){let n=d(e,"a","equal","string_or_numeric"),r=d(t,"b","equal","string_or_numeric");[n,r]=V(n,r),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(Go,s)}const yc=b({equal_:bc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kc(e,t,n){const r=d(t,"a","where"),s=d(n,"b","where"),o=d(e,"condition","where","bool"),i=L(L(o.shape,r.shape),s.shape),a=Se(o,i),c=Se(r,i),u=Se(s,i),h={condition:a,t:c,e:u};return g.runKernel(Lr,h)}const at=b({where_:kc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wc(e){const n={x:d(e,"x","zerosLike")};return g.runKernel(es,n)}const B=b({zerosLike_:wc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xc(e){const n={x:d(e,"x","elu","float32")};return g.runKernel(or,n)}const $c=b({elu_:xc});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vc(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function Sc(e,t,n){const r=e.length+t.length,s=[];let o=0,i=0;for(let a=0;a<r;a++)n.indexOf(a)===-1?s.push(e[o++]):s.push(t[i++]);return s}function Ec(e,t){const n=[],r=e.length;for(let o=0;o<r;o++)t.indexOf(o)===-1&&n.push(e[o]);const s=t.map(o=>e[o]);return[n,s]}function fe(e,t){const n=t.map(r=>1);return Sc(e,n,t)}function ws(e,t){if(vc(e,t))return null;const n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(r=>n.push(r)),n}function tn(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tc(e,t=null,n=!1){const s={x:d(e,"x","max")},o={reductionIndices:t,keepDims:n};return g.runKernel(mr,s,o)}const oe=b({max_:Tc});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ic(e,t=null,n=!1){const s={x:d(e,"x","min")},o={axis:t,keepDims:n};return g.runKernel(xr,s,o)}const En=b({min_:Ic});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nc(e,t){let n=d(e,"base","pow"),r=d(t,"exp","pow");[n,r]=V(n,r);const s={a:n,b:r};return g.runKernel(Dr,s)}const Ht=b({pow_:Nc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function P(e,t){if((ot(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&ot(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Zt(e,[],[],t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dc(e){const n={x:d(e,"x","sqrt","float32")};return g.runKernel(Vr,n)}const nt=b({sqrt_:Dc});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ac(e){const t=d(e,"x","square"),n={};return g.runKernel("Square",{x:t},n)}const R=b({square_:Ac});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cc(e,t=null,n=!1){let r=d(e,"x","sum");r.dtype==="bool"&&(r=S(r,"int32"));const s={x:r},o={axis:t,keepDims:n};return g.runKernel(Hr,s,o)}const I=b({sum_:Cc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fc(e,t="euclidean",n=null,r=!1){e=d(e,"x","norm");const s=xs(e,t,n);let o=s.shape;if(r){const i=ht(n,e.shape);o=fe(s.shape,i)}return x(s,o)}function xs(e,t,n=null){if(e.rank===0)return pt(e);if(e.rank!==1&&n===null)return xs(x(e,[-1]),t,n);if(e.rank===1||typeof n=="number"||Array.isArray(n)&&n.length===1){if(t===1)return I(pt(e),n);if(t===1/0)return oe(pt(e),n);if(t===-1/0)return En(pt(e),n);if(t==="euclidean"||t===2)return nt(I(Ht(pt(e),P(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return oe(I(pt(e),n[0]),n[1]-1);if(t===1/0)return oe(I(pt(e),n[1]),n[0]);if(t===-1/0)return En(I(pt(e),n[1]),n[0]);if(t==="fro"||t==="euclidean")return nt(I(R(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const $s=b({norm_:Fc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bc(e){const n={x:d(e,"x","exp")};return g.runKernel(ar,n)}const Kt=b({exp_:Bc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mc(e,t=0){const n=d(e,"x","expandDims","string_or_numeric");f(t<=n.rank,()=>"Axis must be <= rank of the tensor");const r={input:n},s={dim:t};return g.runKernel(ir,r,s)}const vt=b({expandDims_:Mc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _c(e,t){const n=d(e,"x","tile","string_or_numeric");f(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const r={x:n},s={reps:t};return g.runKernel(je,r,s)}const Wt=b({tile_:_c});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pc(e,t,n,r="float32"){t==null&&(t=e);const s=Qt([e,t],r),o=e<=t?e:t;for(let a=0;a<o;++a)s.set(1,a,a);const i=x(s.toTensor(),[e,t]);if(n==null)return i;if(n.length===1)return Wt(vt(i,0),[n[0],1,1]);if(n.length===2)return Wt(vt(vt(i,0),0),[n[0],n[1],1,1]);if(n.length===3)return Wt(vt(vt(vt(i,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const Gc=b({eye_:Pc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rc(e){const n={x:d(e,"x","floor","float32")};return g.runKernel(cr,n)}const vs=b({floor_:Rc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lc(e,t,n=0,r=0){const s=d(e,"x","gather"),o=d(t,"indices","gather","int32"),i={x:s,indices:o},a={axis:n,batchDims:r};return g.runKernel(hr,i,a)}const Kc=b({gather_:Lc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oc(e,t){let n=d(e,"a","greater","string_or_numeric"),r=d(t,"b","greater","string_or_numeric");[n,r]=V(n,r),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(Oo,s)}const xt=b({greater_:Oc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qc(e,t){let n=d(e,"a","greaterEqual","string_or_numeric"),r=d(t,"b","greaterEqual","string_or_numeric");[n,r]=V(n,r),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(fr,s)}const be=b({greaterEqual_:qc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zc(e){const n={input:d(e,"input","imag")};return g.runKernel(qo,n)}const Uc=b({imag_:zc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wc(e,t=.2){const r={x:d(e,"x","leakyRelu")},s={alpha:t};return g.runKernel(dr,r,s)}const Vc=b({leakyRelu_:Wc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hc(e,t){let n=d(e,"a","less","string_or_numeric"),r=d(t,"b","less","string_or_numeric");[n,r]=V(n,r),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(Vo,s)}const jc=b({less_:Hc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xc(e,t){let n=d(e,"a","lessEqual","string_or_numeric"),r=d(t,"b","lessEqual","string_or_numeric");[n,r]=V(n,r),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(Ho,s)}const te=b({lessEqual_:Xc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yc(e){const n={x:d(e,"x","log","float32")};return g.runKernel(pr,n)}const Ss=b({log_:Yc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jc(e){const n={x:d(e,"x","log1p")};return g.runKernel(gr,n)}const ep=b({log1p_:Jc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zc(e,t){f(Ie(e),()=>"The f passed in variableGrads(f) must be a function"),f(t==null||Array.isArray(t)&&t.every(u=>u instanceof ce),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const u in g.registeredVariables)t.push(g.registeredVariables[u])}const r=n?t.filter(u=>!u.trainable):null,s=t.length;t=t.filter(u=>u.trainable),f(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${s} variables is trainable.`);const o=!0,{value:i,grads:a}=g.gradients(e,t,null,o);f(a.some(u=>u!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),f(i.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${i.rank} tensor`);const c={};return t.forEach((u,h)=>{a[h]!=null&&(c[u.name]=a[h])}),r?.forEach(u=>c[u.name]=null),{value:i,grads:c}}function jt(e){return g.customGrad(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qc(e){const n={x:d(e,"x","neg")};return g.runKernel(Sr,n)}const rt=b({neg_:Qc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tu(e){const n={x:d(e,"x","softplus")};return g.runKernel(Wr,n)}const np=b({softplus_:tu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eu(e,t){let n=d(e,"a","sub"),r=d(t,"b","sub");[n,r]=V(n,r);const s={a:n,b:r};return g.runKernel(Jr,s)}const C=b({sub_:eu});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nu(e,t=-1){const n=d(e,"logits","logSoftmax");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return jt((s,o)=>{const a=oe(s,t,!0),c=C(s,a),u=C(S(c,"float32"),Ss(I(Kt(c),t,!0)));return o([u]),{value:u,gradFunc:(l,p)=>{const[m]=p,k=!0,$=Kt(m);return C(l,y(I(l,t,k),$))}}})(n)}const rp=b({logSoftmax_:nu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ru(e,t){const n=d(e,"a","logicalAnd","bool"),r=d(t,"b","logicalAnd","bool");L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(jo,s)}const en=b({logicalAnd_:ru});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function su(e){const n={x:d(e,"x","logicalNot","bool")};return g.runKernel(Xo,n)}const ou=b({logicalNot_:su});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function au(e,t,n,r,s){const o=d(e,"x","maxPool"),i=1;let a=o,c=!1;o.rank===3&&(c=!0,a=x(o,[1,o.shape[0],o.shape[1],o.shape[2]])),f(a.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${a.rank}.`),f(Nt(n,i),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${i}'`),Q("maxPool",r,s);const u={x:a},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s},l=g.runKernel(yr,u,h);return c?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const sp=b({maxPool_:au});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iu(e,t=[1,1,1],n,r,s,o="NDHWC"){const i=d(e,"x","maxPool3d");let a=i,c=!1;i.rank===4&&(c=!0,a=x(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),f(a.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${a.rank}.`),f(o==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${o}`),Q("maxPool3d",r,s);const u={x:a},h={filterSize:t,strides:n,pad:r,dimRoundingMode:s,dataFormat:o},l=g.runKernel(kr,u,h);return c?x(l,[l.shape[1],l.shape[2],l.shape[3],l.shape[4]]):l}const op=b({maxPool3d_:iu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cu(e,t){let n=d(e,"a","maximum"),r=d(t,"b","maximum");[n,r]=V(n,r),n.dtype==="bool"&&(n=S(n,"int32"),r=S(r,"int32")),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(br,s)}const Es=b({maximum_:cu});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uu(e,t=null,n=!1){const s={x:d(e,"x","mean")},o={axis:t,keepDims:n};return g.runKernel(wr,s,o)}const Tn=b({mean_:uu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xt(e,t="float32"){if(bt(e),t==="complex64"){const r=Xt(e,"float32"),s=Xt(e,"float32");return ge(r,s)}const n=qe(J(e),t);return g.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ye(e,t="float32"){if(bt(e),t==="complex64"){const r=ye(e,"float32"),s=Xt(e,"float32");return ge(r,s)}const n=Ln(J(e),t);return g.makeTensor(n,e,t)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lu(e,t){let n=d(e,"a","minimum"),r=d(t,"b","minimum");[n,r]=V(n,r),n.dtype==="bool"&&(n=S(n,"int32"),r=S(r,"int32")),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel($r,s)}const ap=b({minimum_:lu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hu(e,t=null,n=!1){e=d(e,"x","moments");const r=ht(t,e.shape),s=Tn(e,r,n);let o=s.shape;n||(o=fe(s.shape,r));const i=R(C(S(e,"float32"),x(s,o))),a=Tn(i,r,n);return{mean:s,variance:a}}const ip=b({moments_:hu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fu(e,t){let n=d(e,"a","notEqual","string_or_numeric"),r=d(t,"b","notEqual","string_or_numeric");[n,r]=V(n,r),L(n.shape,r.shape);const s={a:n,b:r};return g.runKernel(ra,s)}const cp=b({notEqual_:fu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function du(e,t,n=1,r=0,s="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const i={indices:d(e,"indices","oneHot","int32")},a={dtype:s,depth:t,onValue:n,offValue:r};return g.runKernel(Tr,i,a)}const up=b({oneHot_:du});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pu(e){const n={x:d(e,"x","onesLike")};return g.runKernel(Er,n)}const lp=b({onesLike_:pu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gu(e,t,n=0){const r=d(e,"x","pad");if(r.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const s={paddings:t,constantValue:n},o={x:r};return g.runKernel(Nr,o,s)}const mu=b({pad_:gu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bu(e,t,n){const r=d(e,"x","spaceToBatchND");f(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),f(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),f(r.shape.reduce((i,a,c)=>c>0&&c<=t.length?i&&(a+n[c-1][0]+n[c-1][1])%t[c-1]===0:i,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const s={x:r},o={blockShape:t,paddings:n};return g.runKernel(jr,s,o)}const yu=b({spaceToBatchND_:bu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ku(e,t){const n=d(e,"x","prelu"),r=d(t,"alpha","prelu"),s={x:n,alpha:r};return g.runKernel(Ar,s)}const wu=b({prelu_:ku});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ts{constructor(t,n,r,s,o){this.mean=t,this.stdDev=n,this.dtype=r,this.nextVal=NaN,this.truncated=s,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const i=o||Math.random();this.random=Gn.alea(i.toString())}nextValue(){if(!isNaN(this.nextVal)){const s=this.nextVal;return this.nextVal=NaN,s}let t,n,r=!1;for(;!r;){let s,o,i;do s=2*this.random()-1,o=2*this.random()-1,i=s*s+o*o;while(i>=1||i===0);const a=Math.sqrt(-2*Math.log(i)/i);t=this.mean+this.stdDev*s*a,n=this.mean+this.stdDev*o*a,(!this.truncated||this.isValidTruncated(t))&&(r=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class xu{constructor(t=0,n=1,r,s){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=n-t,this.dtype=r,s==null&&(s=Math.random()),typeof s=="number"&&(s=s.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${n} <= 1 and dtype is not float`);this.random=Gn.alea(s)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $u(e,t=0,n=1,r,s){if(bt(e),r!=null&&r==="bool")throw new Error(`Unsupported data type ${r}`);const o=new Ts(t,n,r,!1,s),i=Qt(e,r);for(let a=0;a<i.values.length;a++)i.values[a]=o.nextValue();return i.toTensor()}const hp=b({randomNormal_:$u});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vu(e,t=0,n=1,r="float32",s){bt(e);const o=Qt(e,r),i=new xu(t,n,null,s);for(let a=0;a<o.values.length;a++)o.values[a]=i.nextValue();return o.toTensor()}const Su=b({randomUniform_:vu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function de(e,t,n=1,r="float32"){if(n===0)throw new Error("Cannot have a step of zero");const s={start:e,stop:t,step:n,dtype:r};return g.runKernel(ca,{},s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eu(e){const n={input:d(e,"input","real")};return g.runKernel(ua,n)}const Tu=b({real_:Eu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Iu(e){const n={x:d(e,"x","relu")};return g.runKernel(Cr,n)}const Nu=b({relu_:Iu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Du(e){const n={x:d(e,"x","relu6")};return g.runKernel(_r,n)}const Au=b({relu6_:Du});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cu(e,t){const r={x:d(e,"x","reverse")},s={dims:t};return g.runKernel(Pr,r,s)}const Fu=b({reverse_:Cu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bu(e){const n={x:d(e,"x","round")};return g.runKernel(Gr,n)}const Mu=b({round_:Bu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _u(e){const n={x:d(e,"x","rsqrt","float32")};return g.runKernel(Rr,n)}const Pu=b({rsqrt_:_u});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gu(e){const n={x:d(e,"x","selu")};return g.runKernel(Kr,n)}const fp=b({selu_:Gu});function Ru(e,t,n,r,s,o=[1,1],i="NHWC"){const a=d(e,"x","separableConv2d"),c=d(t,"depthwiseFilter","separableConv2d"),u=d(n,"pointwiseFilter","separableConv2d");let h=a,l=!1;if(a.rank===3&&(l=!0,h=x(a,[1,a.shape[0],a.shape[1],a.shape[2]])),i==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");f(h.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${h.rank}.`),f(c.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${c.rank}.`),f(u.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${c.rank}.`),f(u.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${u.shape[0]}.`),f(u.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${u.shape[1]}.`);const p=c.shape[2],m=c.shape[3];f(u.shape[2]===p*m,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${p*m}, but got ${u.shape[2]}.`);const k=mc(h,c,r,s,i,o),w=me(k,u,1,"valid",i);return l?x(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const dp=b({separableConv2d_:Ru});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lu(e){const n={x:d(e,"x","sin","float32")};return g.runKernel(qr,n)}const Ku=b({sin_:Lu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ou(e){const n={x:d(e,"x","sinh")};return g.runKernel(zr,n)}const qu=b({sinh_:Ou});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zu(e,t,n){const r=d(e,"x","slice1d");return f(r.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),K(r,[t],[n])}const pp=b({slice1d_:zu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uu(e,t,n){const r=d(e,"x","slice2d");return f(r.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),K(r,t,n)}const gp=b({slice2d_:Uu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wu(e,t,n){const r=d(e,"x","slice3d");return f(r.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),K(r,t,n)}const mp=b({slice3d_:Wu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vu(e,t,n){const r=d(e,"x","slice4d");return f(r.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),K(r,t,n)}const bp=b({slice4d_:Vu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hu(e,t=-1){const n=d(e,"logits","softmax","float32");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const r={logits:n},s={dim:t};return g.runKernel(Yr,r,s)}const yp=b({softmax_:Hu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ju(e,t,n=0){const s={x:d(e,"x","split")},o={numOrSizeSplits:t,axis:n};return g.runKernel(Xr,s,o)}const nn=b({split_:ju});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xu(e,t){const n=d(e,"x","squeeze","string_or_numeric");return x(n,Hs(n.shape,t).newShape)}const Yu=b({squeeze_:Xu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ju(e,t=0){const n=ls(e,"tensors","stack","string_or_numeric");f(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&f(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const r=n,s={axis:t};return g.runKernel(Ir,r,s)}const Yt=b({stack_:Ju});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zu(e,t=0){const r={x:d(e,"x","step")},s={alpha:t};return g.runKernel(ns,r,s)}const ke=b({step_:Zu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mt(e,t){Re(e);const n=Jt(e,t);if(n.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return Zt(e,null,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ee(e,t,n){if(Re(e),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const r=Jt(e,n);if(r.length!==2&&r.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return Zt(e,t,r,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qu(e,t,n){if(Re(e),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const r=Jt(e,n);if(r.length!==3&&r.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return Zt(e,t,r,n)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tl(e,t=0,n=1,r,s){if(bt(e),r!=null&&r==="bool")throw new Error("Unsupported data type $ { dtype }");const o=new Ts(t,n,r,!0,s),i=Qt(e,r);for(let a=0;a<i.values.length;a++)i.values[a]=o.nextValue();return i.toTensor()}const kp=b({truncatedNormal_:tl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function el(e,t,n){const r=d(e,"x","unsortedSegmentSum"),s=d(t,"segmentIds","unsortedSegmentSum","int32");f(ae(n),()=>"numSegments must be of dtype int");const o={x:r,segmentIds:s},i={numSegments:n};return g.runKernel(ts,o,i)}const nl=b({unsortedSegmentSum_:el});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rl(e,t=0){const n=d(e,"x","unstack","string_or_numeric");f(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const r={value:n},s={axis:t};return g.runKernel(Qr,r,s)}const rn=b({unstack_:rl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wp(e,t=!0,n,r){return g.makeVariable(e,t,n,r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sl(e,t,n){const r=d(e,"x","transpose");if(t==null&&(t=r.shape.map((i,a)=>a).reverse()),f(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(i=>{f(i>=0&&i<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();const s={x:r},o={perm:t};return r.dtype==="complex64"?q(()=>{let i=Tu(r),a=Uc(r);return i=g.runKernel(ne,{x:i},o),a=g.runKernel(ne,{x:a},o),n&&(a=rt(a)),ge(i,a)}):g.runKernel(ne,s,o)}const wt=b({transpose_:sl});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ol(e,t){if(t==null)return e.shape.slice();if(pe(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let r=0;r<e.shape.length;r++)t[r]==null&&e.shape[r]!=null?n.push(e.shape[r]):n.push(t[r]);return n}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function al(e,t,n,r){const s=d(e,"x","dropout");if(f(s.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${s.dtype} tensor instead.`),f(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof tt?s.clone():s;const o=ol(s,n),i=1-t,a=N(vs(E(Su(o,0,1,"float32",r),i)),i);return y(s,a)}const xp=b({dropout_:al});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function il(e,t,n,r,s,o="NHWC",i){let a=e;e.rank===3&&(a=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]])),f(a.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${a.shape}.`),f(c.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${c.shape}.`),f(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const u=o==="NHWC"?a.shape[3]:a.shape[1],h=o==="NHWC"?c.shape[3]:c.shape[1];f(u===n[2],()=>`Error in conv2dDerFilter: depth of input ${u}) must match input depth in filter (${n[2]}.`),f(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),Q("conv2dDerFilter",s,i);const l={x:a,dy:c},p={strides:r,pad:s,dataFormat:o,dimRoundingMode:i,filterShape:n};return g.runKernel(So,l,p)}const sn=b({conv2DBackpropFilter_:il});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Is(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return y(e,ke(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function Ns(e,t){let n=t;const r=U(e.shape,t.shape);return r.length>0&&(n=I(n,r)),x(n,e.shape)}function Ds(e,t,n,r){if(t==="linear")return e;if(t==="relu")return Nu(e);if(t==="elu")return $c(e);if(t==="relu6")return Au(e);if(t==="prelu")return wu(e,n);if(t==="leakyrelu")return Vc(e,r);if(t==="sigmoid")return bs(e);throw new Error(`Unknown fused activation ${t}.`)}const As=(e,t)=>!(e>0)||t==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cl({x:e,filter:t,strides:n,pad:r,dataFormat:s="NHWC",dilations:o=[1,1],dimRoundingMode:i,bias:a,activation:c="linear",preluActivationWeights:u,leakyreluAlpha:h}){if(c=c||"linear",As(g.state.gradientDepth,c)===!1){f(s==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${s} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let D=me(e,t,n,r,s,o,i);return a!=null&&(D=E(D,a)),Ds(D,c,u,h)}const l=d(e,"x","conv2d","float32"),p=d(t,"filter","conv2d","float32");let m=l,k=!1;l.rank===3&&(k=!0,m=x(l,[1,l.shape[0],l.shape[1],l.shape[2]])),f(m.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${m.rank}.`),f(p.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${p.rank}.`),Q("fused conv2d",r,i);const $=s==="NHWC"?m.shape[3]:m.shape[1];f(p.shape[2]===$,()=>`Error in conv2d: depth of input (${$}) must match input depth for filter ${p.shape[2]}.`),f(Nt(n,o),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`);const w=Ni(m.shape,p.shape,n,o,r,i);let v;a!=null&&(v=d(a,"bias","fused conv2d"),[v]=V(v,l),s==="NHWC"?L(w.outShape,v.shape):(f(v.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${v.shape.length}.`),f(v.shape.length===0||v.shape[0]===w.outChannels||v.shape[0]===1,()=>`Error in fused conv2d: bias shape (${v.shape}) is not compatible with the number of output channels (${w.outChannels})`)));let A;if(u!=null){const D=u.shape;if(f(D.length<=1||D.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${D.length}.`),D.length===1)f(D[0]===1||D[0]===w.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${D}) is not compatible with the number of output channels (${w.outChannels}).`);else if(D.length===3)try{L(D,w.outShape)}catch{const O=`Error in fused conv2d: PReLU activation weights (${D}) is not compatible with the output shape of the conv2d (${w.outShape}).`;throw Error(O)}A=d(u,"prelu weights","fused conv2d")}const M=(D,j)=>{f(s==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${s} but only NHWC is currently supported.`);const[O,H,X,W]=j,ft=Is(D,X,c);f(Rt(o),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${o}'`);const it=Qe(H.shape,ft,O,n,r),ct=sn(H,ft,O.shape,n,r),ut=[it,ct];if(W!=null){const At=Ns(W,ft);ut.push(At)}return ut},T={x:m,filter:p,bias:v,preluActivationWeights:A},_={strides:n,pad:r,dataFormat:s,dilations:o,dimRoundingMode:i,activation:c,leakyreluAlpha:h};return a==null?jt((j,O,H)=>{let X=g.runKernel(dn,T,_);return H([O,j,X]),k&&(X=x(X,[X.shape[1],X.shape[2],X.shape[3]])),{value:X,gradFunc:M}})(m,p):jt((j,O,H,X)=>{let W=g.runKernel(dn,T,_);return X([O,j,W,H]),k&&(W=x(W,[W.shape[1],W.shape[2],W.shape[3]])),{value:W,gradFunc:M}})(m,p,v)}const $p=b({fusedConv2d_:cl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ul(e,t,n,r,s,o=[1,1],i){let a=e;e.rank===3&&(a=x(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let c=t;c.rank===3&&(c=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const u={x:a,dy:c},h={strides:r,pad:s,dimRoundingMode:i,dilations:o,filterShape:n};return g.runKernel(Ao,u,h)}const ll=b({depthwiseConv2dNativeBackpropFilter_:ul});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hl(e,t,n,r,s,o=[1,1],i){let a=t,c=!1;t.rank===3&&(c=!0,a=x(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const u={dy:a,filter:n},h={strides:r,pad:s,dimRoundingMode:i,dilations:o,inputShape:e},l=g.runKernel(Co,u,h);return c?x(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const fl=b({depthwiseConv2dNativeBackpropInput_:hl});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dl({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:s,activation:o="linear",preluActivationWeights:i,leakyreluAlpha:a=.2}){if(As(g.state.gradientDepth,o)===!1){let W=G(e,t,n,r);return s!=null&&(W=E(W,s)),Ds(W,o,i,a)}let c=d(e,"a","fused matMul"),u=d(t,"b","fused matMul");[c,u]=V(c,u);const h=n?c.shape[c.rank-2]:c.shape[c.rank-1],l=r?u.shape[u.rank-1]:u.shape[u.rank-2],p=n?c.shape[c.rank-1]:c.shape[c.rank-2],m=r?u.shape[u.rank-2]:u.shape[u.rank-1],k=c.shape.slice(0,-2),$=u.shape.slice(0,-2),w=J(k),v=J($);f(h===l,()=>`Error in fused matMul: inner shapes (${h}) and (${l}) of Tensors with shapes ${c.shape} and ${u.shape} and transposeA=${n} and transposeB=${r} must match.`);const M=L(c.shape.slice(0,-2),u.shape.slice(0,-2)).concat([p,m]),T=n?x(c,[w,h,p]):x(c,[w,p,h]),_=r?x(u,[v,m,l]):x(u,[v,l,m]);let D;s!=null&&(D=d(s,"bias","fused matMul"),[D]=V(D,c),L(M,D.shape));let j;i!=null&&(j=d(i,"prelu weights","fused matMul"));const O=(W,ft)=>{const[it,ct,ut,At]=ft,dt=Is(x(W,ut.shape),ut,o);let Ct,Ft;if(!n&&!r?(Ct=G(dt,ct,!1,!0),Ft=G(it,dt,!0,!1)):!n&&r?(Ct=G(dt,ct,!1,!1),Ft=G(dt,it,!0,!1)):n&&!r?(Ct=G(ct,dt,!1,!0),Ft=G(it,dt,!1,!1)):(Ct=G(ct,dt,!0,!0),Ft=G(dt,it,!0,!0)),s!=null){const Os=Ns(At,dt);return[Ct,Ft,Os]}else return[Ct,Ft]},H={a:T,b:_,bias:D,preluActivationWeights:j},X={transposeA:n,transposeB:r,activation:o,leakyreluAlpha:a};return s==null?jt((ft,it,ct)=>{const ut=g.runKernel(fn,H,X);return ct([ft,it,ut]),{value:x(ut,M),gradFunc:O}})(T,_):jt((ft,it,ct,ut)=>{const At=g.runKernel(fn,H,X);return ut([ft,it,At,ct]),{value:x(At,M),gradFunc:O}})(T,_,D)}const vp=b({fusedMatMul_:dl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pl(e,t,n,r,s="bilinear",o=0){const i=d(e,"image","cropAndResize"),a=d(t,"boxes","cropAndResize","float32"),c=d(n,"boxInd","cropAndResize","int32"),u=a.shape[0];f(i.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${i.rank}.`),f(a.rank===2&&a.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${u},4] but had shape ${a.shape}.`),f(c.rank===1&&c.shape[0]===u,()=>`Error in cropAndResize: boxInd must be have size [${u}] but had shape ${a.shape}.`),f(r.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),f(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),f(s==="bilinear"||s==="nearest",()=>`method must be bilinear or nearest, but was ${s}`);const h={image:i,boxes:a,boxInd:c},l={method:s,extrapolationValue:o,cropSize:r};return g.runKernel(No,h,l)}const gl=b({cropAndResize_:pl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ml(e){const t=d(e,"image","flipLeftRight","float32");f(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return g.runKernel(Ko,n,{})}const bl=b({flipLeftRight_:ml});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yl(e){const t=d(e,"image","grayscaleToRGB"),n=t.rank-1,r=t.shape[n];f(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),f(r===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);const s=new Array(t.rank);return s.fill(1,0,n),s[n]=3,Wt(t,s)}const kl=b({grayscaleToRGB_:yl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wl(e,t,n=0,r=.5){const s=d(e,"image","rotateWithOffset","float32");f(s.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${s.rank}.`);const o={image:s},i={radians:t,fillValue:n,center:r};return g.runKernel(ya,o,i)}const xl=b({rotateWithOffset_:wl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ot(e,t,n,r,s,o){r==null&&(r=.5),s==null&&(s=Number.NEGATIVE_INFINITY),o==null&&(o=0);const i=e.shape[0];return n=Math.min(n,i),f(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),f(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),f(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),f(t.rank===1,()=>"scores must be a 1D tensor"),f(t.shape[0]===i,()=>`scores has incompatible shape with boxes. Expected ${i}, but was ${t.shape[0]}`),f(0<=o&&o<=1,()=>`softNmsSigma must be in [0, 1], but was '${o}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:s,softNmsSigma:o}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $l(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY){const o=d(e,"boxes","nonMaxSuppression","float32"),i=d(t,"scores","nonMaxSuppression","float32"),a=Ot(o,i,n,r,s);n=a.maxOutputSize,r=a.iouThreshold,s=a.scoreThreshold;const c={maxOutputSize:n,iouThreshold:r,scoreThreshold:s};return g.runKernel(sa,{boxes:o,scores:i},c)}const vl=b({nonMaxSuppression_:$l});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sl(e,t,n){const r=El(e,t,n),s=r<0?-(r+1):r;e.splice(s,0,t)}function El(e,t,n){return Il(e,t,n||Tl)}function Tl(e,t){return e>t?1:e<t?-1:0}function Il(e,t,n){let r=0,s=e.length,o=0,i=!1;for(;r<s;){o=r+(s-r>>>1);const a=n(t,e[o]);a>0?r=o+1:(s=o,i=!a)}return i?r:-r-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nl(e,t,n,r,s){return on(e,t,n,r,s,0)}function Dl(e,t,n,r,s,o){return on(e,t,n,r,s,0,!1,o,!0)}function Al(e,t,n,r,s,o){return on(e,t,n,r,s,o,!0)}function on(e,t,n,r,s,o,i=!1,a=!1,c=!1){const u=[];for(let w=0;w<t.length;w++)t[w]>s&&u.push({score:t[w],boxIndex:w,suppressBeginIndex:0});u.sort(In);const h=o>0?-.5/o:0,l=[],p=[];for(;l.length<n&&u.length>0;){const w=u.pop(),{score:v,boxIndex:A,suppressBeginIndex:M}=w;if(v<s)break;let T=!1;for(let _=l.length-1;_>=M;--_){const D=Cl(e,A,l[_]);if(D>=r){T=!0;break}if(w.score=w.score*Fl(r,h,D),w.score<=s)break}w.suppressBeginIndex=l.length,T||(w.score===v?(l.push(A),p.push(w.score)):w.score>s&&Sl(u,w,In))}const m=l.length,k=n-m;a&&k>0&&(l.push(...new Array(k).fill(0)),p.push(...new Array(k).fill(0)));const $={selectedIndices:l};return i&&($.selectedScores=p),c&&($.validOutputs=m),$}function Cl(e,t,n){const r=e.subarray(t*4,t*4+4),s=e.subarray(n*4,n*4+4),o=Math.min(r[0],r[2]),i=Math.min(r[1],r[3]),a=Math.max(r[0],r[2]),c=Math.max(r[1],r[3]),u=Math.min(s[0],s[2]),h=Math.min(s[1],s[3]),l=Math.max(s[0],s[2]),p=Math.max(s[1],s[3]),m=(a-o)*(c-i),k=(l-u)*(p-h);if(m<=0||k<=0)return 0;const $=Math.max(o,u),w=Math.max(i,h),v=Math.min(a,l),A=Math.min(c,p),M=Math.max(v-$,0)*Math.max(A-w,0);return M/(m+k-M)}function Fl(e,t,n){const r=Math.exp(t*n*n);return n<=e?r:0}function In(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Bl(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY){const o=d(e,"boxes","nonMaxSuppressionAsync"),i=d(t,"scores","nonMaxSuppressionAsync"),a=Ot(o,i,n,r,s);n=a.maxOutputSize,r=a.iouThreshold,s=a.scoreThreshold;const c=await Promise.all([o.data(),i.data()]),u=c[0],h=c[1],{selectedIndices:l}=Nl(u,h,n,r,s);return o!==e&&o.dispose(),i!==t&&i.dispose(),mt(l,"int32")}const Ml=Bl;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _l(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,o=0){const i=d(e,"boxes","nonMaxSuppression"),a=d(t,"scores","nonMaxSuppression"),c=Ot(i,a,n,r,s,o);n=c.maxOutputSize,r=c.iouThreshold,s=c.scoreThreshold,o=c.softNmsSigma;const u={boxes:i,scores:a},h={maxOutputSize:n,iouThreshold:r,scoreThreshold:s,softNmsSigma:o},l=g.runKernel(aa,u,h);return{selectedIndices:l[0],selectedScores:l[1]}}const Pl=b({nonMaxSuppressionWithScore_:_l});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Gl(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,o=0){const i=d(e,"boxes","nonMaxSuppressionAsync"),a=d(t,"scores","nonMaxSuppressionAsync"),c=Ot(i,a,n,r,s,o);n=c.maxOutputSize,r=c.iouThreshold,s=c.scoreThreshold,o=c.softNmsSigma;const u=await Promise.all([i.data(),a.data()]),h=u[0],l=u[1],{selectedIndices:p,selectedScores:m}=Al(h,l,n,r,s,o);return i!==e&&i.dispose(),a!==t&&a.dispose(),{selectedIndices:mt(p,"int32"),selectedScores:mt(m)}}const Rl=Gl;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ll(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,o=!1){const i=d(e,"boxes","nonMaxSuppression"),a=d(t,"scores","nonMaxSuppression"),c=Ot(i,a,n,r,s,null),u=c.maxOutputSize,h=c.iouThreshold,l=c.scoreThreshold,p={boxes:i,scores:a},m={maxOutputSize:u,iouThreshold:h,scoreThreshold:l,padToMaxOutputSize:o},k=g.runKernel(oa,p,m);return{selectedIndices:k[0],validOutputs:k[1]}}const Kl=b({nonMaxSuppressionPadded_:Ll});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ol(e,t,n,r=.5,s=Number.NEGATIVE_INFINITY,o=!1){const i=d(e,"boxes","nonMaxSuppressionAsync"),a=d(t,"scores","nonMaxSuppressionAsync"),c=Ot(i,a,n,r,s,null),u=c.maxOutputSize,h=c.iouThreshold,l=c.scoreThreshold,[p,m]=await Promise.all([i.data(),a.data()]),{selectedIndices:k,validOutputs:$}=Dl(p,m,u,h,l,o);return i!==e&&i.dispose(),a!==t&&a.dispose(),{selectedIndices:mt(k,"int32"),validOutputs:P($,"int32")}}const ql=Ol;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zl(e,t,n=!1,r=!1){const s=d(e,"images","resizeBilinear");f(s.rank===3||s.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${s.rank}.`),f(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),f(r===!1||n===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let o=s,i=!1;s.rank===3&&(i=!0,o=x(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const a={images:o},c={alignCorners:n,halfPixelCenters:r,size:t},u=g.runKernel(Mr,a,c);return i?x(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Ul=b({resizeBilinear_:zl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wl(e,t,n=!1,r=!1){const s=d(e,"images","resizeNearestNeighbor");f(s.rank===3||s.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${s.rank}.`),f(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),f(s.dtype==="float32"||s.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),f(r===!1||n===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let o=s,i=!1;s.rank===3&&(i=!0,o=x(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const a={images:o},c={alignCorners:n,halfPixelCenters:r,size:t},u=g.runKernel(Br,a,c);return i?x(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Vl=b({resizeNearestNeighbor_:Wl});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hl(e,t="binary",n=!1,r=.5){const s=d(e,"image","threshold"),o=.2989,i=.587,a=.114,c=s.shape[0]*s.shape[1];let u=y(mt([r]),255),h,l,p,m;if(f(s.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${s.rank}.`),f(s.shape[2]===3||s.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${s.shape[2]}.`),f(s.dtype==="int32"||s.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${s.dtype}.`),f(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),s.shape[2]===3){[h,l,p]=nn(s,[1,1,1],-1);const w=y(h,o),v=y(l,i),A=y(p,a);m=E(E(w,v),A)}else m=e;if(t==="otsu"){const w=ji(S(Mu(m),"int32"),se([]),256);u=jl(w,c)}const k=n?te(m,u):xt(m,u);return S(y(k,255),"int32")}function jl(e,t){let n=mt([-1]),r=mt([0]),s=mt([0]),o,i,a,c,u,h;for(let l=0;l<e.size-1;l++){o=K(e,0,l+1),i=K(e,l+1),u=N(I(o),t),h=N(I(i),t);const p=I(y(o,de(0,o.size)));a=N(p,I(o));const m=Ze(i.shape,o.size),k=E(de(0,i.size),m),$=y(i,k);c=N(I($),I(i));const w=C(a,c),v=C(a,c),A=y(u,h);s=y(y(A,w),v);const M=xt(s,r);r=at(M,s,r),n=at(M,mt([l]),n)}return n}const Xl=b({threshold_:Hl});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yl(e,t,n="nearest",r="constant",s=0,o){const i=d(e,"image","transform","float32"),a=d(t,"transforms","transform","float32");f(i.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${i.rank}.`),f(a.rank===2&&(a.shape[0]===i.shape[0]||a.shape[0]===1)&&a.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),f(o==null||o.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${o}.`);const c={image:i,transforms:a},u={interpolation:n,fillMode:r,fillValue:s,outputShape:o};return g.runKernel(ba,c,u)}const Jl=b({transform_:Yl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zl(e,t,n){f(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),f(n%1===0,()=>`bandPart(): numUpper must be an integer, got ${n}.`);const r=d(e,"a","bandPart");f(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);const s=r.shape,[o,i]=r.shape.slice(-2);if(!(t<=o))throw new Error(`bandPart(): numLower (${t}) must not be greater than the number of rows (${o}).`);if(!(n<=i))throw new Error(`bandPart(): numUpper (${n}) must not be greater than the number of columns (${i}).`);t<0&&(t=o),n<0&&(n=i);const a=x(de(0,o,1,"int32"),[-1,1]),c=de(0,i,1,"int32"),u=C(a,c),h=en(te(u,P(+t,"int32")),be(u,P(-n,"int32"))),l=Xt([o,i],r.dtype);return x(Yt(rn(x(r,[-1,o,i])).map(p=>at(h,p,l))),s)}const Ql=b({bandPart_:Zl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function th(e){let t;if(Array.isArray(e)){t=!1,f(e!=null&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const s=e[0].shape[0];for(let o=1;o<e.length;++o)f(e[o].shape[0]===s,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[o].shape[0]} vs. ${s})`)}else t=!0,e=nn(e,e.shape[0],0).map(s=>Yu(s,[0]));f(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],r=e;for(let s=0;s<e.length;++s)n.push(g.tidy(()=>{let o=r[s];if(s>0)for(let i=0;i<s;++i){const a=y(I(y(n[i],o)),n[i]);o=C(o,a)}return N(o,$s(o,"euclidean"))}));return t?Yt(n,0):n}const eh=b({gramSchmidt_:th});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nh(e,t=!1){if(f(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return Nn(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((c,u)=>c*u),r=rn(x(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),s=[],o=[];r.forEach(c=>{const[u,h]=Nn(c,t);s.push(u),o.push(h)});const i=x(Yt(s,0),e.shape),a=x(Yt(o,0),e.shape);return[i,a]}}function Nn(e,t=!1){return g.tidy(()=>{f(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],r=e.shape[1];let s=Gc(n),o=Pt(e);const i=Ee([[1]],[1,1]);let a=Pt(i);const c=n>=r?r:n;for(let u=0;u<c;++u){const h=o,l=a,p=s;[a,o,s]=g.tidy(()=>{const m=K(o,[u,u],[n-u,1]),k=$s(m),$=K(o,[u,u],[1,1]),w=at(xt($,0),Ee([[-1]]),Ee([[1]])),v=C($,y(w,k)),A=N(m,v);A.shape[0]===1?a=Pt(i):a=kt([i,K(A,[1,0],[A.shape[0]-1,A.shape[1]])],0);const M=rt(N(G(w,v),k)),T=K(o,[u,0],[n-u,r]),_=y(M,a),D=wt(a);if(u===0)o=C(T,G(_,G(D,T)));else{const H=C(T,G(_,G(D,T)));o=kt([K(o,[0,0],[u,r]),H],0)}const j=wt(_),O=K(s,[0,u],[n,s.shape[1]-u]);if(u===0)s=C(O,G(G(O,a),j));else{const H=C(O,G(G(O,a),j));s=kt([K(s,[0,0],[n,u]),H],1)}return[a,o,s]}),Z([h,l,p])}return!t&&n>r&&(s=K(s,[0,0],[n,r]),o=K(o,[0,0],[r,r])),[s,o]})}const rh=b({qr_:nh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sp={flipLeftRight:bl,grayscaleToRGB:kl,resizeNearestNeighbor:Vl,resizeBilinear:Ul,rotateWithOffset:xl,cropAndResize:gl,nonMaxSuppression:vl,nonMaxSuppressionAsync:Ml,nonMaxSuppressionWithScore:Pl,nonMaxSuppressionWithScoreAsync:Rl,nonMaxSuppressionPadded:Kl,nonMaxSuppressionPaddedAsync:ql,threshold:Xl,transform:Jl},Ep={bandPart:Ql,gramSchmidt:eh,qr:rh};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class sh{getClassName(){return this.constructor.className}static fromConfig(t,n){return new t(n)}}class St{constructor(){this.classNameMap={}}static getMap(){return St.instance==null&&(St.instance=new St),St.instance}static register(t){St.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function oh(e){f(e.className!=null,()=>"Class being registered does not have the static className property defined."),f(typeof e.className=="string",()=>"className is required to be a string, but got type "+typeof e.className),f(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),St.register(e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Dt extends sh{minimize(t,n=!1,r){const{value:s,grads:o}=this.computeGradients(t,r);if(r!=null){const i=r.map(a=>({name:a.name,tensor:o[a.name]}));this.applyGradients(i)}else this.applyGradients(o);return Z(o),n?s:(s.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,n){return Zc(t,n)}dispose(){this.iterations_!=null&&Z(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:P(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(Dt,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Cs extends Dt{constructor(t,n,r=null){super(),this.learningRate=t,this.rho=n,this.epsilon=r,this.accumulatedGrads=[],this.accumulatedUpdates=[],r==null&&(this.epsilon=g.backend.epsilon())}static get className(){return"Adadelta"}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=g.registeredVariables[r],i=!1;this.accumulatedGrads[s]==null&&(this.accumulatedGrads[s]={originalName:`${r}/accum_grad`,variable:q(()=>B(o).variable(i))}),this.accumulatedUpdates[s]==null&&(this.accumulatedUpdates[s]={originalName:`${r}/accum_var`,variable:q(()=>B(o).variable(i))});const a=Array.isArray(t)?t[s].tensor:t[r];if(a==null)return;const c=this.accumulatedGrads[s].variable,u=this.accumulatedUpdates[s].variable;q(()=>{const h=E(y(c,this.rho),y(R(a),1-this.rho)),l=y(N(nt(E(u,this.epsilon)),nt(E(c,this.epsilon))),a),p=E(y(u,this.rho),y(R(l),1-this.rho));c.assign(h),u.assign(p);const m=E(y(l,-this.learningRate),o);o.assign(m)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(Z(this.accumulatedGrads.map(t=>t.variable)),Z(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=t.length/2,r=!1;this.accumulatedGrads=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedUpdates=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.rho,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Fs extends Dt{constructor(t,n=.1){super(),this.learningRate=t,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}static get className(){return"Adagrad"}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=g.registeredVariables[r];this.accumulatedGrads[s]==null&&(this.accumulatedGrads[s]={originalName:`${r}/accumulator`,variable:q(()=>Ze(o.shape,this.initialAccumulatorValue).variable(!1))});const i=Array.isArray(t)?t[s].tensor:t[r];if(i==null)return;const a=this.accumulatedGrads[s].variable;q(()=>{const c=E(a,R(i));a.assign(c);const u=E(y(N(i,nt(E(c,g.backend.epsilon()))),-this.learningRate),o);o.assign(u)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&Z(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulatedGrads=t.map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,n){return new t(n.learningRate,n.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bs extends Dt{constructor(t,n,r,s=null){super(),this.learningRate=t,this.beta1=n,this.beta2=r,this.epsilon=s,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],q(()=>{this.accBeta1=P(n).variable(),this.accBeta2=P(r).variable()}),s==null&&(this.epsilon=g.backend.epsilon())}static get className(){return"Adam"}applyGradients(t){const n=Array.isArray(t)?t.map(r=>r.name):Object.keys(t);q(()=>{const r=C(1,this.accBeta1),s=C(1,this.accBeta2);n.forEach((o,i)=>{const a=g.registeredVariables[o],c=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:q(()=>B(a).variable(c))}),this.accumulatedSecondMoment[i]==null&&(this.accumulatedSecondMoment[i]={originalName:`${o}/v`,variable:q(()=>B(a).variable(c))});const u=Array.isArray(t)?t[i].tensor:t[o];if(u==null)return;const h=this.accumulatedFirstMoment[i].variable,l=this.accumulatedSecondMoment[i].variable,p=E(y(h,this.beta1),y(u,1-this.beta1)),m=E(y(l,this.beta2),y(R(u),1-this.beta2)),k=N(p,r),$=N(m,s);h.assign(p),l.assign(m);const w=E(y(N(k,E(nt($),this.epsilon)),-this.learningRate),a);a.assign(w)}),this.accBeta1.assign(y(this.accBeta1,this.beta1)),this.accBeta2.assign(y(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&Z(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&Z(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t),q(()=>{this.accBeta1.assign(Ht(this.beta1,this.iterations_+1)),this.accBeta2.assign(Ht(this.beta2,this.iterations_+1))});const n=t.length/2,r=!1;this.accumulatedFirstMoment=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedSecondMoment=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ms extends Dt{constructor(t,n,r,s=null,o=0){super(),this.learningRate=t,this.beta1=n,this.beta2=r,this.epsilon=s,this.decay=o,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],q(()=>{this.iteration=P(0).variable(),this.accBeta1=P(n).variable()}),s==null&&(this.epsilon=g.backend.epsilon())}static get className(){return"Adamax"}applyGradients(t){const n=Array.isArray(t)?t.map(r=>r.name):Object.keys(t);q(()=>{const r=C(1,this.accBeta1),s=N(-this.learningRate,E(y(this.iteration,this.decay),1));n.forEach((o,i)=>{const a=g.registeredVariables[o],c=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:B(a).variable(c)}),this.accumulatedWeightedInfNorm[i]==null&&(this.accumulatedWeightedInfNorm[i]={originalName:`${o}/v`,variable:B(a).variable(c)});const u=Array.isArray(t)?t[i].tensor:t[o];if(u==null)return;const h=this.accumulatedFirstMoment[i].variable,l=this.accumulatedWeightedInfNorm[i].variable,p=E(y(h,this.beta1),y(u,1-this.beta1)),m=y(l,this.beta2),k=pt(u),$=Es(m,k);h.assign(p),l.assign($);const w=E(y(N(s,r),N(p,E($,this.epsilon))),a);a.assign(w)}),this.iteration.assign(E(this.iteration,1)),this.accBeta1.assign(y(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&Z(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&Z(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class an extends Dt{constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}static get className(){return"SGD"}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=Array.isArray(t)?t[s].tensor:t[r];if(o==null)return;const i=g.registeredVariables[r];q(()=>{const a=E(y(this.c,o),i);i.assign(a)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=yi(P(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,n){return new t(n.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class _s extends an{constructor(t,n,r=!1){super(t),this.learningRate=t,this.momentum=n,this.useNesterov=r,this.accumulations=[],this.m=P(this.momentum)}static get className(){return"Momentum"}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=g.registeredVariables[r];this.accumulations[s]==null&&(this.accumulations[s]={originalName:`${r}/momentum`,variable:q(()=>B(o).variable(!1))});const i=this.accumulations[s].variable,a=Array.isArray(t)?t[s].tensor:t[r];a!=null&&q(()=>{let c;const u=E(y(this.m,i),a);this.useNesterov?c=E(y(this.c,E(a,y(u,this.m))),o):c=E(y(this.c,u),o),i.assign(u),o.assign(c)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&Z(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulations=t.map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,n){return new t(n.learningRate,n.momentum,n.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ps extends Dt{constructor(t,n=.9,r=0,s=null,o=!1){if(super(),this.learningRate=t,this.decay=n,this.momentum=r,this.epsilon=s,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=o,s==null&&(this.epsilon=g.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}static get className(){return"RMSProp"}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=g.registeredVariables[r],i=!1;this.accumulatedMeanSquares[s]==null&&(this.accumulatedMeanSquares[s]={originalName:`${r}/rms`,variable:q(()=>B(o).variable(i))}),this.accumulatedMoments[s]==null&&(this.accumulatedMoments[s]={originalName:`${r}/momentum`,variable:q(()=>B(o).variable(i))}),this.accumulatedMeanGrads[s]==null&&this.centered&&(this.accumulatedMeanGrads[s]={originalName:`${r}/mg`,variable:q(()=>B(o).variable(i))});const a=Array.isArray(t)?t[s].tensor:t[r];if(a==null)return;const c=this.accumulatedMeanSquares[s].variable,u=this.accumulatedMoments[s].variable;q(()=>{const h=E(y(c,this.decay),y(R(a),1-this.decay));if(this.centered){const l=this.accumulatedMeanGrads[s].variable,p=E(y(l,this.decay),y(a,1-this.decay)),m=N(y(a,this.learningRate),nt(C(h,E(R(p),this.epsilon)))),k=E(y(u,this.momentum),m);c.assign(h),l.assign(p),u.assign(k);const $=C(o,k);o.assign($)}else{const l=E(y(c,this.decay),y(R(a),1-this.decay)),p=E(y(u,this.momentum),N(y(a,this.learningRate),nt(E(l,this.epsilon))));c.assign(l),u.assign(p);const m=C(o,p);o.assign(m)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&Z(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&Z(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&Z(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=this.centered?t.length/3:t.length/2,r=!1;this.accumulatedMeanSquares=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedMoments=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.centered&&(this.accumulatedMeanGrads=t.slice(n*2,n*3).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,n){return new t(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ah=[Cs,Fs,Bs,Ms,_s,Ps,an];function ih(){for(const e of ah)oh(e)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dn(e,t,n,r){i(e),n=n??0,r=r??1,a(n,r);let s=0;const o=c=>(c.then(u=>{const h=n+ ++s/e.length*(r-n);return t(h),u}),c);function i(c){f(c!=null&&Array.isArray(c)&&c.length>0,()=>"promises must be a none empty array")}function a(c,u){f(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${c}`),f(u>=0&&u<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${u}`),f(u>=c,()=>`startFraction must be no more than endFraction, but got startFraction ${c} and endFraction ${u}`)}return Promise.all(e.map(o))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function ch(e,t){t==null&&(t={});const n=t.fetchFunc==null?F().platform.fetch:t.fetchFunc,r=e.map(l=>n(l,t.requestInit,{isBinary:!0})),s=0,o=.5,a=(t.onProgress==null?await Promise.all(r):await Dn(r,t.onProgress,s,o)).map(l=>l.arrayBuffer()),c=.5,u=1;return t.onProgress==null?await Promise.all(a):await Dn(a,t.onProgress,c,u)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uh="application/octet-stream",lh="application/json";class cn{constructor(t,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.onProgress=n.onProgress,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(f(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=F().platform.fetch,f(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&f(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{}}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const r=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],s=Va(t,r);n.body.append("model.json",new Blob([JSON.stringify(s)],{type:lh}),"model.json"),t.weightData!=null&&n.body.append("model.weights.bin",new Blob([t.weightData],{type:uh}),"model.weights.bin");const o=await this.fetch(this.path,n);if(o.ok)return{modelArtifactsInfo:Ye(t),responses:[o]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${o.status}.`)}async load(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await t.json()}catch{let i=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?i+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":i+=" Please make sure the server is serving valid JSON for this request.",new Error(i)}const r=n.modelTopology,s=n.weightsManifest;if(r==null&&s==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return ja(n,o=>this.loadWeights(o))}async loadWeights(t){const n=Array.isArray(this.path)?this.path[1]:this.path,[r,s]=hh(n),o=this.weightPathPrefix||r,i=Xa(t),a=[],c=[];for(const h of t)for(const l of h.paths)this.weightUrlConverter!=null?c.push(this.weightUrlConverter(l)):a.push(o+l+s);this.weightUrlConverter&&a.push(...await Promise.all(c));const u=await ch(a,{requestInit:this.requestInit,fetchFunc:this.fetch,onProgress:this.onProgress});return[i,Wa(u)]}}cn.URL_SCHEME_REGEX=/^https?:\/\//;function hh(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),r=e.substring(0,t),s=n>t?e.substring(n):"";return[r+"/",s]}function An(e){return e.match(cn.URL_SCHEME_REGEX)!=null}const Gs=(e,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(Array.isArray(e)?n=e.every(r=>An(r)):n=An(e),n)return Rs(e,t)}return null};z.registerSaveRouter(Gs);z.registerLoadRouter(Gs);function Rs(e,t){return new cn(e,t)}function Tp(e,t){return Rs(e,t)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let $t;function fh(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(e==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,r=!1,s=!1,o=!1,i=!1,a=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<"u"&&e instanceof ImageData)r=!0;else if(typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement)s=!0;else if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement)o=!0;else if(e.getContext!=null)i=!0;else if(typeof ImageBitmap<"u"&&e instanceof ImageBitmap)a=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(De(hn,g.backendName)!=null){const k={pixels:e},$={numChannels:t};return g.runKernel(hn,k,$)}const[u,h]=s?[e.videoWidth,e.videoHeight]:[e.width,e.height];let l;if(i)l=e.getContext("2d").getImageData(0,0,u,h).data;else if(r||n)l=e.data;else if(o||s||a){if($t==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")$t=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else $t=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});$t.canvas.width=u,$t.canvas.height=h,$t.drawImage(e,0,0,u,h),l=$t.getImageData(0,0,u,h).data}let p;if(t===4)p=new Int32Array(l);else{const k=u*h;p=new Int32Array(k*t);for(let $=0;$<k;$++)for(let w=0;w<t;++w)p[$*t+w]=l[$*4+w]}return Qu(p,[h,u,t],"int32")}function dh(e){return e!=null&&e.data instanceof Uint8Array}function ph(){return typeof window<"u"&&typeof ImageBitmap<"u"&&window.hasOwnProperty("createImageBitmap")}function gh(e){return e!=null&&e.width!==0&&e.height!==0}function mh(e){return ph()&&!(e instanceof ImageBitmap)&&gh(e)&&!dh(e)}async function Ip(e,t=3){let n=null;if(F().getBool("WRAP_TO_IMAGEBITMAP")&&mh(e)){let r;try{r=await createImageBitmap(e,{premultiplyAlpha:"none"})}catch{r=null}r!=null&&r.width===e.width&&r.height===e.height?n=r:n=e}else n=e;return fh(n,t)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bh(e,t,n){let r;const s=e.shape.length;typeof t=="number"?r=[t,...new Array(s-1).fill(0)]:t.length<s?r=t.concat(new Array(s-t.length).fill(0)):r=t.slice(),r.forEach(i=>{f(i!==-1,()=>"slice() does not support negative begin indexing.")});let o;return n==null?o=new Array(s).fill(-1):typeof n=="number"?o=[n,...new Array(s-1).fill(-1)]:n.length<s?o=n.concat(new Array(s-n.length).fill(-1)):o=n,o=o.map((i,a)=>i>=0?i:(f(i===-1,()=>`Negative size values should be exactly -1 but got ${i} for the slice() size at index ${a}.`),e.shape[a]-r[a])),[r,o]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class yh{static sgd(t){return new an(t)}static momentum(t,n,r=!1){return new _s(t,n,r)}static rmsprop(t,n=.9,r=0,s=null,o=!1){return new Ps(t,n,r,s,o)}static adam(t=.001,n=.9,r=.999,s=null){return new Bs(t,n,r,s)}static adadelta(t=.001,n=.95,r=null){return new Cs(t,n,r)}static adamax(t=.002,n=.9,r=.999,s=null,o=0){return new Ms(t,n,r,s,o)}static adagrad(t,n=.1){return new Fs(t,n)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Np=yh;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kh=(()=>typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:e=>e())();function Dp(){return new Promise(e=>kh(()=>e()))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wh=1.7580993408473768,xh=1.0507009873554805;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ih();/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ls={kernelName:qn,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(e,ke(S(n,"float32"),-1))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $h={kernelName:ao,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>{const r=R(S(n,"float32")),s=nt(C(P(1),r));return rt(N(e,s))}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vh={kernelName:io,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>{const r=nt(C(R(S(n,"float32")),1));return N(e,r)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sh={kernelName:We,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=L(n.shape,r.shape);return{a:()=>{let a=e;const c=U(n.shape,s);return c.length>0&&(a=I(a,c)),x(a,n.shape)},b:()=>{let a=e;const c=U(r.shape,s);return c.length>0&&(a=I(a,c)),x(a,r.shape)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Eh={kernelName:co,saveAllInputs:!0,gradFunc:(e,t)=>{const n={};return t.forEach((r,s)=>{n[s]=()=>e.clone()}),n}};/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Th={kernelName:zn,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>B(n)}}};/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ih={kernelName:ho,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>B(n)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nh={kernelName:fo,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,nt(C(P(1),R(S(n,"float32")))))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dh={kernelName:po,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>{const r=nt(E(P(1),R(S(n,"float32"))));return N(e,r)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ah={kernelName:bo,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=L(n.shape,r.shape);return{a:()=>{const a=E(R(n),R(r));let c=y(e,N(r,a));const u=U(n.shape,s);return u.length>0&&(c=I(c,u)),x(c,n.shape)},b:()=>{const a=E(R(n),R(r));let c=rt(y(e,N(n,a)));const u=U(r.shape,s);return u.length>0&&(c=I(c,u)),x(c,r.shape)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ch={kernelName:go,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,E(R(S(n,"float32")),1))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fh={kernelName:mo,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,C(P(1),R(S(n,"float32"))))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bh(e,t,n,r,s,o){const i=d(e,"dy","avgPool3dGrad"),a=d(t,"input","avgPool3dGrad");let c=i,u=a,h=!1;a.rank===4&&(h=!0,c=x(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]]),u=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),f(c.rank===5,()=>`Error in avgPool3dGrad: dy must be rank 5 but got rank ${c.rank}.`),f(u.rank===5,()=>`Error in avgPool3dGrad: input must be rank 5 but got rank ${u.rank}.`),Q("avgPool3dGrad",s,o);const l={dy:c,input:u},p={filterSize:n,strides:r,pad:s,dimRoundingMode:o},m=g.runKernel(ko,l,p);return h?x(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}const Mh=b({avgPool3dGrad_:Bh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _h={kernelName:Wn,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{filterSize:s,strides:o,pad:i,dimRoundingMode:a}=n;return{x:()=>Mh(e,r,s,o,i,a)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ph(e,t,n,r,s){const o=d(e,"dy","avgPoolGrad"),i=d(t,"input","avgPoolGrad");f(i.rank===o.rank,()=>`Rank of input (${i.rank}) does not match rank of dy (${o.rank})`);let a=i,c=o,u=!1;i.rank===3&&(u=!0,a=x(i,[1,i.shape[0],i.shape[1],i.shape[2]]),c=x(o,[1,o.shape[0],o.shape[1],o.shape[2]])),f(c.rank===4,()=>`Error in avgPoolGrad: dy must be rank 4 but got rank ${c.rank}.`),f(a.rank===4,()=>`Error in avgPoolGrad: input must be rank 4 but got rank ${a.rank}.`);const h={dy:c,input:a},l={filterSize:n,strides:r,pad:s},p=g.runKernel(yo,h,l);return u?x(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const Gh=b({avgPoolGrad_:Ph});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rh={kernelName:Un,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{filterSize:s,strides:o,pad:i}=n;return{x:()=>Gh(e,r,s,o,i)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lh={kernelName:Vn,inputsToSave:["a","b"],gradFunc:(e,t,n)=>{const[r,s]=t,{transposeA:o,transposeB:i}=n;return!o&&!i?{a:()=>G(e,s,!1,!0),b:()=>G(r,e,!0,!1)}:!o&&i?{a:()=>G(e,s,!1,!1),b:()=>G(e,r,!0,!1)}:o&&!i?{a:()=>G(s,e,!1,!0),b:()=>G(r,e,!1,!1)}:{a:()=>G(s,e,!0,!0),b:()=>G(e,r,!0,!0)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kh={kernelName:Hn,gradFunc:(e,t,n)=>{const{blockShape:r,crops:s}=n;return{x:()=>yu(e,r,s)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oh={kernelName:xo,gradFunc:(e,t,n)=>{const r=n,s=r.inputShape,o=r.shape,i=Array.from(o);for(let c=s.length-1;c>=0;c--)if(s[c]===o[c])i[c]=1;else if(s[c]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${o}].`);const a=[];for(let c=0;c<i.length;c++)i[c]>1&&a.push(c);return{x:()=>I(e,a,!0)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qh={kernelName:Ve,gradFunc:e=>({x:()=>e.clone()})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zh={kernelName:$o,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uh={kernelName:jn,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{clipValueMin:s,clipValueMax:o}=n;return{x:()=>at(en(be(r,s),te(r,o)),e,B(e))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wh={kernelName:Xn,inputsToSave:["x"],gradFunc:Ls.gradFunc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vh={kernelName:Yn,saveAllInputs:!0,gradFunc:(e,t,n)=>{const r=t.map(c=>c.shape),{axis:s}=n,o=ht(s,t[0].shape)[0],i=r.map(c=>c[o]);return nn(e,i,o).map(c=>()=>c)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hh={kernelName:Jn,inputsToSave:["x","filter"],gradFunc:(e,t,n)=>{const[r,s]=t,{dilations:o,strides:i,pad:a,dataFormat:c}=n;return f(Rt(o),()=>`Error in gradient of conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${o}'`),{x:()=>Qe(r.shape,e,s,i,a,c),filter:()=>sn(r,e,s.shape,i,a,c)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jh={kernelName:Zn,inputsToSave:["dy","filter"],gradFunc:(e,t,n)=>{const[r,s]=t,{strides:o,pad:i,dataFormat:a,dimRoundingMode:c}=n;return{dy:()=>me(e,s,o,i,a,1,c),filter:()=>sn(e,r,s.shape,o,i,a,c)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xh(e,t,n,r,s){let o=e;e.rank===4&&(o=x(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]));let i=t;i.rank===4&&(i=x(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]])),f(o.rank===5,()=>`Error in conv3dDerFilter: input must be rank 5, but got shape ${o.shape}.`),f(i.rank===5,()=>`Error in conv3dDerFilter: dy must be rank 5, but got shape ${i.shape}.`),f(n.length===5,()=>`Error in conv3dDerFilter: filterShape must be length 5, but got ${n}.`),f(o.shape[4]===n[3],()=>`Error in conv3dDerFilter: depth of input ${o.shape[4]}) must match input depth in filter (${n[3]}.`),f(i.shape[4]===n[4],()=>`Error in conv3dDerFilter: depth of dy (${i.shape[4]}) must match output depth for filter (${n[4]}).`);const a={x:o,dy:i},c={strides:r,pad:s,filterShape:n};return g.runKernel(Eo,a,c)}const Yh=b({conv3DBackpropFilter_:Xh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jh={kernelName:Qn,inputsToSave:["x","filter"],gradFunc:(e,t,n)=>{const{dilations:r,strides:s,pad:o}=n;f(Rt(r),()=>`Error in gradient of conv3D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${r}'`);const[i,a]=t;return{x:()=>ys(i.shape,e,a,s,o),filter:()=>Yh(i,e,a.shape,s,o)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zh={kernelName:tr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(rt(Ku(S(n,"float32"))),e)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qh={kernelName:er,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(qu(S(n,"float32")),e)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tf={kernelName:nr,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{axis:s,exclusive:o,reverse:i}=n;return{x:()=>{const a=ws([s],r.rank);let c=dc(e,s,o,!i);return a!=null&&(c=wt(c,a)),c}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ef={kernelName:rr,inputsToSave:["x","filter"],gradFunc:(e,t,n)=>{const{dilations:r,strides:s,pad:o,dimRoundingMode:i}=n,a=r??[1,1];f(Rt(a),()=>`Error in gradient of depthwiseConv2dNative: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[c,u]=t;return f(c.rank===4,()=>`Error in gradient of depthwiseConv2dNative: input must be rank 4, but got rank ${c.rank}.`),f(u.rank===4,()=>`Error in gradient of depthwiseConv2dNative: filter must be rank 4, but got rank ${u.rank}.`),f(c.shape[3]===u.shape[2],()=>`Error in gradient of depthwiseConv2d: number of input channels (${c.shape[3]}) must match the inChannels dimension in filter ${u.shape[2]}.`),f(Nt(s,a),()=>`Error in gradient of depthwiseConv2d: Either strides or dilations must be  1. Got strides ${s} and dilations '${a}'.`),Q("depthwiseConv2d",o,i),{x:()=>fl(c.shape,e,u,s,o,a,i),filter:()=>ll(c,e,u.shape,s,o,a,i)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nf={kernelName:Fo,inputsToSave:["x","filter"],gradFunc:(e,t,n)=>{const[r,s]=t,o={x:r,filter:s,dy:e},i={x:r,filter:s,dy:e};return{x:()=>g.runKernel(Bo,o,n),filter:()=>g.runKernel(Mo,i,n)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rf={kernelName:or,outputsToSave:[!0],gradFunc:(e,t)=>{const[n]=t,r={dy:e,y:n};return{x:()=>g.runKernel(_o,r)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sf={kernelName:Po,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t,r=y(Kt(rt(R(n))),2/Math.sqrt(Math.PI));return{x:()=>y(e,r)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const of={kernelName:ar,outputsToSave:[!0],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(e,n)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const af={kernelName:ir,inputsToSave:["input"],gradFunc:(e,t)=>{const[n]=t;return{input:()=>x(e,n.shape)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cf={kernelName:Ro,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(e,Kt(n))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uf={kernelName:cr,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lf={kernelName:ur,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=L(n.shape,r.shape);return{a:()=>{const a=N(e,S(r,"float32")),c=U(n.shape,s);return c.length>0?x(I(a,c),n.shape):a},b:()=>{let a=y(e,S(n,"float32"));const c=U(r.shape,s);c.length>0&&(a=x(I(a,c),r.shape));const u=R(r);return rt(N(a,S(u,"float32")))}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hf={kernelName:lr,inputsToSave:["x","mean","variance","scale"],gradFunc:(e,t,n)=>{const{varianceEpsilon:r}=n,[s,o,i,a]=t,c=a??P(1),u=U(o.shape,s.shape),h=[];if(o.rank===1){for(let T=0;T<s.shape.length-1;++T)h.push(s.shape[T]);h.push(1)}const l=C(s,o),p=y(e,c),m=Pu(E(i,P(r))),k=y(y(y(m,m),m),P(-.5));return{x:()=>o.rank===1?x(y(y(e,Wt(x(m,[1,1,1,o.shape[0]]),h)),c),s.shape):x(y(y(e,m),c),s.shape),mean:()=>{let T=y(y(m,P(-1)),p);return o.rank===1&&(T=I(T,u)),x(T,o.shape)},variance:()=>{let T=y(y(k,l),p);return o.rank===1&&(T=I(T,u)),x(T,o.shape)},scale:()=>{const T=y(l,m);let _=y(e,T);return o.rank===1&&(_=I(_,u)),x(_,o.shape)},offset:()=>{let T=e;return o.rank===1&&(T=I(T,u)),x(T,o.shape)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ff={kernelName:hr,inputsToSave:["x","indices"],gradFunc:(e,t,n)=>{const[r,s]=t,{axis:o}=n,i=ht(o,r.shape)[0];return{x:()=>{const c=r.shape,u=s.size,h=c.slice(0,i),l=h.length,p=c.slice(o,c.length).slice(1),m=p.length,k=Cn(0,l),$=Cn(l+1,l+1+m),w=Fn([h,[u],p]),v=x(e,w),A=x(s,[u]),M=Fn([[l],k,$]),T=wt(v,M);let _=nl(T,A,r.shape[i]);const D=tn(M);return _=wt(_,D),_},indices:()=>s}}};function Cn(e,t){const n=[];for(let r=e;r<t;++r)n.push(r);return n}function Fn(e){const t=[];for(let n=0;n<e.length;++n)for(let r=0;r<e[n].length;++r)t.push(e[n][r]);return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const df={kernelName:fr,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t;return{a:()=>B(n),b:()=>B(r)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pf={kernelName:He,gradFunc:e=>({x:()=>S(e,"float32")})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gf={kernelName:zo,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mf={kernelName:Uo,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bf={kernelName:Wo,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yf={kernelName:dr,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{alpha:s}=n,o=xt(r,0);return{x:()=>at(o,e,y(e,s))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kf={kernelName:gr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,E(n,1))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wf={kernelName:pr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,S(n,"float32"))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xf={kernelName:Yo,inputsToSave:[],outputsToSave:[!0],gradFunc:(e,t,n)=>{const[r]=t,{axis:s}=n;return{logits:()=>{const i=Kt(r);return C(e,y(I(e,s,!0),i))}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $f(e,t,n,r=5,s=1,o=1,i=.5){const a={x:e,y:t,dy:n},c={depthRadius:r,bias:s,alpha:o,beta:i};return g.runKernel(Zo,a,c)}const vf=b({localResponseNormalizationBackprop_:$f});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sf={kernelName:Jo,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,n)=>{const[r,s]=t,{depthRadius:o,bias:i,alpha:a,beta:c}=n;return{x:()=>vf(r,s,e,o,i,a,c)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ks(e,t,n,r){return t.rank<n.rank&&(t=x(t,fe(t.shape,r))),e.rank<n.rank&&(e=x(e,fe(e.shape,r))),{x:()=>y(e,S(yc(n,t),e.dtype))}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bn={kernelName:mr,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,n)=>{const r=n,{reductionIndices:s}=r,o=t[0],i=t[1],a=ht(s,o.shape),c=Ks(e,i,o,a);return{x:()=>c.x()}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ef={kernelName:br,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t;return{a:()=>y(e,S(be(n,r),"float32")),b:()=>y(e,S(jc(n,r),"float32"))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tf(e,t,n,r,s,o,i){const a=d(e,"dy","maxPool3dGrad"),c=d(t,"input","maxPool3dGrad"),u=d(n,"output","maxPool3dGrad");let h=a,l=c,p=u,m=!1;c.rank===4&&(m=!0,h=x(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]]),l=x(c,[1,c.shape[0],c.shape[1],c.shape[2],c.shape[3]]),p=x(u,[1,u.shape[0],u.shape[1],u.shape[2],u.shape[3]])),f(h.rank===5,()=>`Error in maxPool3dGrad: dy must be rank 5 but got rank ${h.rank}.`),f(l.rank===5,()=>`Error in maxPool3dGrad: input must be rank 5 but got rank ${l.rank}.`),f(p.rank===5,()=>`Error in maxPool3dGrad: output must be rank 5 but got rank ${p.rank}.`),Q("maxPool3dGrad",o,i);const k={dy:h,input:l,output:p},$={filterSize:r,strides:s,pad:o,dimRoundingMode:i},w=g.runKernel(ta,k,$);return m?x(w,[w.shape[1],w.shape[2],w.shape[3],w.shape[4]]):w}const If=b({maxPool3dGrad_:Tf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nf={kernelName:kr,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,n)=>{const[r,s]=t,{filterSize:o,strides:i,pad:a,dimRoundingMode:c}=n;return{x:()=>If(e,r,s,o,i,a,c)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Df(e,t,n,r,s,o,i){const a=d(e,"dy","maxPoolGrad"),c=d(t,"input","maxPoolGrad"),u=d(n,"output","maxPoolGrad");f(c.rank===a.rank,()=>`Rank of input (${c.rank}) does not match rank of dy (${a.rank})`),f(a.rank===4,()=>`Error in maxPoolGrad: dy must be rank 4 but got rank ${a.rank}.`),f(c.rank===4,()=>`Error in maxPoolGrad: input must be rank 4 but got rank ${c.rank}.`),Q("maxPoolGrad",o,i);const h={dy:a,input:c,output:u},l={filterSize:r,strides:s,pad:o,dimRoundingMode:i};return g.runKernel(Qo,h,l)}const Af=b({maxPoolGrad_:Df});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cf={kernelName:yr,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,n)=>{const[r,s]=t,{filterSize:o,strides:i,pad:a}=n;return{x:()=>Af(e,r,s,o,i,a)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ff={kernelName:wr,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{axis:s}=n,o=ht(s,r.shape),a=Ec(r.shape,o)[1],c=J(a);return{x:()=>{const h=r.shape.slice();o.forEach(m=>{h[m]=1});const l=x(e,h);return N(y(l,ye(r.shape,"float32")),c)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bf={kernelName:xr,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,n)=>{const r=n,{axis:s}=r,[o,i]=t,a=ht(s,o.shape),c=Ks(e,i,o,a);return{x:()=>c.x()}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mf={kernelName:$r,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t;return{a:()=>y(e,S(te(n,r),"float32")),b:()=>y(e,S(xt(n,r),"float32"))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _f={kernelName:ea,inputsToSave:["x"],gradFunc:(e,t,n)=>{const r=t[0],{paddings:s}=n,o=s.map(i=>i[0]);return{x:()=>K(e,o,r.shape)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pf={kernelName:na,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=L(n.shape,r.shape);return{a:()=>{const a=U(n.shape,s);return a.length>0?x(I(e,a),n.shape):e},b:()=>{const a=y(e,rt(vs(N(n,r)))),c=U(r.shape,s);return c.length>0?x(I(a,c),r.shape):a}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gf={kernelName:vr,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=L(n.shape,r.shape);return{a:()=>{const a=y(e,S(r,"float32")),c=U(n.shape,s);return c.length>0?x(I(a,c),n.shape):a},b:()=>{const a=y(e,S(n,"float32")),c=U(r.shape,s);return c.length>0?x(I(a,c),r.shape):a}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rf={kernelName:Sr,gradFunc:e=>({x:()=>rt(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lf={kernelName:Tr,inputsToSave:["indices"],gradFunc:(e,t)=>{const n=t[0];return{indices:()=>Xt(n.shape,"float32")}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kf={kernelName:Er,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Of={kernelName:Ir,saveAllInputs:!0,gradFunc:(e,t,n)=>{const{axis:r}=n;return rn(e,r).map(o=>()=>o)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mn={kernelName:Nr,inputsToSave:["x"],gradFunc:(e,t,n)=>{const r=t[0],{paddings:s}=n,o=s.map(i=>i[0]);return{x:()=>K(e,o,r.shape)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qf={kernelName:Dr,inputsToSave:["a","b"],outputsToSave:[!0],gradFunc:(e,t)=>{const[n,r,s]=t,o=n,i=r,a=L(o.shape,i.shape);return{a:()=>{const h=S(i,"float32");let l=y(e,y(h,Ht(o,C(h,P(1)))));const p=U(o.shape,a);return p.length>0&&(l=I(l,p)),x(l,o.shape)},b:()=>{const h=xt(o,0),l=at(h,Ss(o),B(o));let p=y(e,y(s,l));const m=U(i.shape,a);return m.length>0&&(p=I(p,m)),x(p,i.shape)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zf={kernelName:Ar,inputsToSave:["x","alpha"],gradFunc:(e,t)=>{const[n,r]=t,s=xt(n,0);return{x:()=>at(s,e,y(e,r)),alpha:()=>{let o=at(s,B(e),y(e,n));const i=U(r.shape,e.shape);return i.length>0&&(o=I(o,i)),x(o,r.shape)}}}};/**
 * @license
 * Copyright 2022 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uf(e,t,n){const r=e.shape.slice();r[n]=1;const s=x(t,r),o=Sn(e,n,!0,!1),i=Sn(e,n,!0,!0),a=y(o,i);return y(s,a)}function Wf(e,t,n){const r=e.shape.length,s=r-n.length,o=ws(n,r);let i=e;o!=null&&(i=wt(e,o));const a=i.shape.slice(),u=a.splice(r-n.length,n.length).reduce((p,m)=>p*m,1);a.push(u);const h=i.reshape(a);let l=Uf(h,t,s);if(l=l.reshape(i.shape),o!=null){const p=tn(o);l=wt(l,p)}return l}const Vf={kernelName:ia,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{axis:s}=n;let o=[];return s==null?o=r.shape.map((i,a)=>a):typeof s=="number"?o=[s]:o=s,{x:()=>Wf(r,e,o)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hf={kernelName:sr,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=L(n.shape,r.shape);return{a:()=>{const a=N(e,S(r,"float32")),c=U(n.shape,s);return c.length>0?x(I(a,c),n.shape):a},b:()=>{let a=y(e,S(n,"float32"));const c=U(r.shape,s);c.length>0&&(a=x(I(a,c),r.shape));const u=R(r);return rt(N(a,S(u,"float32")))}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jf={kernelName:la,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,rt(R(n)))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xf={kernelName:_r,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t,r=y(te(n,6),ke(n));return{x:()=>y(e,S(r,"float32"))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yf={kernelName:Cr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(e,S(ke(n),"float32"))}}};/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jf={kernelName:Fr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>x(e,n.shape)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zf={kernelName:Mr,inputsToSave:["images"],gradFunc:(e,t,n)=>{const[r]=t,s={dy:e,images:r};return{images:()=>g.runKernel(fa,s,n)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qf={kernelName:Br,inputsToSave:["images"],gradFunc:(e,t,n)=>{const[r]=t,s={dy:e,images:r};return{images:()=>g.runKernel(ha,s,n)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const td={kernelName:Pr,gradFunc:(e,t,n)=>{const{dims:r}=n,s=ht(r,e.shape);return{x:()=>Fu(e,s)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ed={kernelName:Gr,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nd={kernelName:Rr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>rt(N(e,y(Ht(n,1.5),2)))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rd={kernelName:Lr,inputsToSave:["condition"],gradFunc:(e,t)=>{const[n]=t;return{condition:()=>S(B(n),"float32"),t:()=>y(e,S(n,e.dtype)),e:()=>y(e,S(ou(n),e.dtype))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sd={kernelName:Kr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>{const r=xt(n,P(0)),s=P(wh),o=P(xh),i=y(e,o),a=y(y(e,s),Kt(S(n,"float32")));return at(r,i,a)}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const od={kernelName:Ur,outputsToSave:[!0],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(e,y(n,C(P(1),n)))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ad={kernelName:da,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const id={kernelName:qr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(ks(S(n,"float32")),e)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cd={kernelName:zr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(lc(S(n,"float32")),e)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ud={kernelName:Or,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{begin:s,size:o}=n,i=r.shape,[a,c]=bh(r,s,o),u=[];for(let h=0;h<e.rank;h++)u.push([a[h],i[h]-a[h]-c[h]]);return{x:()=>mu(e,u)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ld={kernelName:Yr,outputsToSave:[!0],gradFunc:(e,t,n)=>{const[r]=t,{dim:s}=n,o=!0,i=y(e,r);return{logits:()=>C(i,y(I(i,[s],o),r))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hd={kernelName:Wr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(e,bs(n))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _n={kernelName:jr,gradFunc:(e,t,n)=>{const{blockShape:r,paddings:s}=n;return{x:()=>Oi(e,r,s)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pn={kernelName:Xr,gradFunc:(e,t,n)=>{const{axis:r}=n;return{x:()=>kt(e,r)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fd={kernelName:Vr,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,y(nt(S(n,"float32")),2))}}};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dd={kernelName:ga,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(e,y(S(n,"float32"),2))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pd={kernelName:pa,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=P(2);return{a:()=>y(e,y(s,C(n,r))),b:()=>y(e,y(s,C(r,n)))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gd={kernelName:ns,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const md={kernelName:Jr,inputsToSave:["a","b"],gradFunc:(e,t)=>{const[n,r]=t,s=L(n.shape,r.shape);return{a:()=>{let a=e;const c=U(n.shape,s);return c.length>0&&(a=I(a,c)),x(a,n.shape)},b:()=>{let a=e;const c=U(r.shape,s);return c.length>0&&(a=I(a,c)),x(rt(a),r.shape)}}}};/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bd={kernelName:Hr,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,s=r.shape.slice(),{axis:o}=n;ht(o,r.shape).forEach(u=>{s[u]=1});const a=x(e,s),c=y(a,ye(r.shape,"float32"));return{x:()=>c}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yd={kernelName:ma,inputsToSave:["x"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>N(e,R(ks(n)))}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kd={kernelName:Zr,outputsToSave:[!0],gradFunc:(e,t)=>{const[n]=t;return{x:()=>y(C(P(1),R(n)),e)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wd={kernelName:je,inputsToSave:["x"],gradFunc:(e,t,n)=>{const[r]=t,{reps:s}=n;return{x:()=>{let i=B(r);if(r.rank===1)for(let a=0;a<s[0];++a)i=E(i,K(e,[a*r.shape[0]],[r.shape[0]]));else if(r.rank===2)for(let a=0;a<s[0];++a)for(let c=0;c<s[1];++c)i=E(i,K(e,[a*r.shape[0],c*r.shape[1]],[r.shape[0],r.shape[1]]));else if(r.rank===3)for(let a=0;a<s[0];++a)for(let c=0;c<s[1];++c)for(let u=0;u<s[2];++u)i=E(i,K(e,[a*r.shape[0],c*r.shape[1],u*r.shape[2]],[r.shape[0],r.shape[1],r.shape[2]]));else if(r.rank===4)for(let a=0;a<s[0];++a)for(let c=0;c<s[1];++c)for(let u=0;u<s[2];++u)for(let h=0;h<s[3];++h)i=E(i,K(e,[a*r.shape[0],c*r.shape[1],u*r.shape[2],h*r.shape[3]],[r.shape[0],r.shape[1],r.shape[2],r.shape[3]]));else throw new Error(`Gradient for tile operation is not implemented for rank-${r.rank} tensors yet.`);return i}}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xd={kernelName:ne,gradFunc:(e,t,n)=>{const r=n,{perm:s}=r,o=tn(s);return{x:()=>wt(e,o)}}};/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $d={kernelName:Qr,gradFunc:(e,t,n)=>{const r=n,{axis:s}=r;return{value:()=>Yt(e,s)}}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vd={kernelName:ts,inputsToSave:["segmentIds"],gradFunc:(e,t)=>{const[n]=t;return{x:()=>Sd(e,n)}}};function Sd(e,t){const n=Es(t,B(t)),r=Kc(e,n);let s=be(t,P(0,"int32"));const o=r.rank-s.rank;for(let a=0;a<o;++a)s=vt(s,a+1);s=en(s,ye(r.shape,"bool"));const i=B(r);return at(s,r,i)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ed={kernelName:es,gradFunc:e=>({x:()=>B(e)})};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Td=[Ls,$h,vh,Sh,Eh,Th,Ih,Nh,Dh,Ah,Ch,Fh,_h,Rh,Lh,Kh,Oh,qh,zh,Uh,Wh,Vh,jh,Hh,Jh,Zh,Qh,tf,ef,nf,Hf,rf,sf,of,af,cf,lf,uf,hf,ff,df,pf,gf,mf,bf,yf,kf,wf,xf,Sf,Bn,Bn,Ef,Nf,Cf,Ff,Bf,Mf,_f,Pf,Gf,Rf,Lf,Kf,Of,Mn,Mn,qf,zf,Vf,jf,Xf,Yf,Jf,Zf,Qf,td,ed,nd,rd,sd,od,ad,id,cd,ud,ld,hd,_n,_n,Pn,Pn,fd,pd,dd,gd,md,bd,yd,kd,wd,xd,$d,vd,Ed];for(const e of Td)ka(e);export{Es as $,Hd as A,Vd as B,xp as C,oh as D,Xt as E,ye as F,P as G,Su as H,kp as I,Gc as J,Ep as K,St as L,wp as M,pe as N,Z as O,Md as P,F as Q,Nu as R,sh as S,nt as T,I as U,yi as V,Dp as W,Qs as X,vs as Y,up as Z,Ze as _,f as a,se as a$,yp as a0,rt as a1,Ss as a2,C as a3,Tn as a4,oe as a5,np as a6,ep as a7,Kt as a8,lp as a9,dp as aA,Sp as aB,Yd as aC,$p as aD,Zd as aE,mc as aF,nn as aG,vt as aH,Fu as aI,rn as aJ,Yt as aK,me as aL,Gd as aM,cp as aN,B as aO,Pd as aP,G as aQ,be as aR,ip as aS,qd as aT,zd as aU,Ud as aV,mu as aW,sp as aX,Ld as aY,op as aZ,Kd as a_,xt as aa,yc as ab,Rd as ac,Yu as ad,at as ae,en as af,Np as ag,Pt as ah,tt as ai,Dt as aj,Dd as ak,Fd as al,Ad as am,Wa as an,Bd as ao,Tp as ap,Cd as aq,fp as ar,ap as as,bs as at,Od as au,rp as av,Vc as aw,wu as ax,Jd as ay,Qd as az,_d as b,de as b0,tp as b1,En as b2,Ip as b3,E as c,S as d,$c as e,bp as f,mp as g,gp as h,pp as i,kt as j,Wt as k,hp as l,vp as m,ie as n,wt as o,mt as p,Kc as q,x as r,K as s,q as t,y as u,N as v,pt as w,Wd as x,Xd as y,jd as z};
