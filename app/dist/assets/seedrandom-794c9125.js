import{c as E,g as U}from"./long-edff4021.js";var D={},V={get exports(){return D},set exports(x){D=x}};(function(x){(function(j,s,c){function p(n){var e=this,i=u();e.next=function(){var t=2091639*e.s0+e.c*23283064365386963e-26;return e.s0=e.s1,e.s1=e.s2,e.s2=t-(e.c=t|0)},e.c=1,e.s0=i(" "),e.s1=i(" "),e.s2=i(" "),e.s0-=i(n),e.s0<0&&(e.s0+=1),e.s1-=i(n),e.s1<0&&(e.s1+=1),e.s2-=i(n),e.s2<0&&(e.s2+=1),i=null}function h(n,e){return e.c=n.c,e.s0=n.s0,e.s1=n.s1,e.s2=n.s2,e}function v(n,e){var i=new p(n),t=e&&e.state,r=i.next;return r.int32=function(){return i.next()*4294967296|0},r.double=function(){return r()+(r()*2097152|0)*11102230246251565e-32},r.quick=r,t&&(typeof t=="object"&&h(t,i),r.state=function(){return h(i,{})}),r}function u(){var n=4022871197,e=function(i){i=String(i);for(var t=0;t<i.length;t++){n+=i.charCodeAt(t);var r=.02519603282416938*n;n=r>>>0,r-=n,r*=n,n=r>>>0,r-=n,n+=r*4294967296}return(n>>>0)*23283064365386963e-26};return e}s&&s.exports?s.exports=v:c&&c.amd?c(function(){return v}):this.alea=v})(E,x,!1)})(V);var O={},F={get exports(){return O},set exports(x){O=x}};(function(x){(function(j,s,c){function p(u){var n=this,e="";n.x=0,n.y=0,n.z=0,n.w=0,n.next=function(){var t=n.x^n.x<<11;return n.x=n.y,n.y=n.z,n.z=n.w,n.w^=n.w>>>19^t^t>>>8},u===(u|0)?n.x=u:e+=u;for(var i=0;i<e.length+64;i++)n.x^=e.charCodeAt(i)|0,n.next()}function h(u,n){return n.x=u.x,n.y=u.y,n.z=u.z,n.w=u.w,n}function v(u,n){var e=new p(u),i=n&&n.state,t=function(){return(e.next()>>>0)/4294967296};return t.double=function(){do var r=e.next()>>>11,o=(e.next()>>>0)/4294967296,a=(r+o)/(1<<21);while(a===0);return a},t.int32=e.next,t.quick=t,i&&(typeof i=="object"&&h(i,e),t.state=function(){return h(e,{})}),t}s&&s.exports?s.exports=v:c&&c.amd?c(function(){return v}):this.xor128=v})(E,x,!1)})(F);var R={},H={get exports(){return R},set exports(x){R=x}};(function(x){(function(j,s,c){function p(u){var n=this,e="";n.next=function(){var t=n.x^n.x>>>2;return n.x=n.y,n.y=n.z,n.z=n.w,n.w=n.v,(n.d=n.d+362437|0)+(n.v=n.v^n.v<<4^(t^t<<1))|0},n.x=0,n.y=0,n.z=0,n.w=0,n.v=0,u===(u|0)?n.x=u:e+=u;for(var i=0;i<e.length+64;i++)n.x^=e.charCodeAt(i)|0,i==e.length&&(n.d=n.x<<10^n.x>>>4),n.next()}function h(u,n){return n.x=u.x,n.y=u.y,n.z=u.z,n.w=u.w,n.v=u.v,n.d=u.d,n}function v(u,n){var e=new p(u),i=n&&n.state,t=function(){return(e.next()>>>0)/4294967296};return t.double=function(){do var r=e.next()>>>11,o=(e.next()>>>0)/4294967296,a=(r+o)/(1<<21);while(a===0);return a},t.int32=e.next,t.quick=t,i&&(typeof i=="object"&&h(i,e),t.state=function(){return h(e,{})}),t}s&&s.exports?s.exports=v:c&&c.amd?c(function(){return v}):this.xorwow=v})(E,x,!1)})(H);var k={},I={get exports(){return k},set exports(x){k=x}};(function(x){(function(j,s,c){function p(u){var n=this;n.next=function(){var i=n.x,t=n.i,r,o;return r=i[t],r^=r>>>7,o=r^r<<24,r=i[t+1&7],o^=r^r>>>10,r=i[t+3&7],o^=r^r>>>3,r=i[t+4&7],o^=r^r<<7,r=i[t+7&7],r=r^r<<13,o^=r^r<<9,i[t]=o,n.i=t+1&7,o};function e(i,t){var r,o=[];if(t===(t|0))o[0]=t;else for(t=""+t,r=0;r<t.length;++r)o[r&7]=o[r&7]<<15^t.charCodeAt(r)+o[r+1&7]<<13;for(;o.length<8;)o.push(0);for(r=0;r<8&&o[r]===0;++r);for(r==8?o[7]=-1:o[r],i.x=o,i.i=0,r=256;r>0;--r)i.next()}e(n,u)}function h(u,n){return n.x=u.x.slice(),n.i=u.i,n}function v(u,n){u==null&&(u=+new Date);var e=new p(u),i=n&&n.state,t=function(){return(e.next()>>>0)/4294967296};return t.double=function(){do var r=e.next()>>>11,o=(e.next()>>>0)/4294967296,a=(r+o)/(1<<21);while(a===0);return a},t.int32=e.next,t.quick=t,i&&(i.x&&h(i,e),t.state=function(){return h(e,{})}),t}s&&s.exports?s.exports=v:c&&c.amd?c(function(){return v}):this.xorshift7=v})(E,x,!1)})(I);var N={},J={get exports(){return N},set exports(x){N=x}};(function(x){(function(j,s,c){function p(u){var n=this;n.next=function(){var i=n.w,t=n.X,r=n.i,o,a;return n.w=i=i+1640531527|0,a=t[r+34&127],o=t[r=r+1&127],a^=a<<13,o^=o<<17,a^=a>>>15,o^=o>>>12,a=t[r]=a^o,n.i=r,a+(i^i>>>16)|0};function e(i,t){var r,o,a,b,S,z=[],_=128;for(t===(t|0)?(o=t,t=null):(t=t+"\0",o=0,_=Math.max(_,t.length)),a=0,b=-32;b<_;++b)t&&(o^=t.charCodeAt((b+32)%t.length)),b===0&&(S=o),o^=o<<10,o^=o>>>15,o^=o<<4,o^=o>>>13,b>=0&&(S=S+1640531527|0,r=z[b&127]^=o+S,a=r==0?a+1:0);for(a>=128&&(z[(t&&t.length||0)&127]=-1),a=127,b=4*128;b>0;--b)o=z[a+34&127],r=z[a=a+1&127],o^=o<<13,r^=r<<17,o^=o>>>15,r^=r>>>12,z[a]=o^r;i.w=S,i.X=z,i.i=a}e(n,u)}function h(u,n){return n.i=u.i,n.w=u.w,n.X=u.X.slice(),n}function v(u,n){u==null&&(u=+new Date);var e=new p(u),i=n&&n.state,t=function(){return(e.next()>>>0)/4294967296};return t.double=function(){do var r=e.next()>>>11,o=(e.next()>>>0)/4294967296,a=(r+o)/(1<<21);while(a===0);return a},t.int32=e.next,t.quick=t,i&&(i.X&&h(i,e),t.state=function(){return h(e,{})}),t}s&&s.exports?s.exports=v:c&&c.amd?c(function(){return v}):this.xor4096=v})(E,x,!1)})(J);var P={},K={get exports(){return P},set exports(x){P=x}};(function(x){(function(j,s,c){function p(u){var n=this,e="";n.next=function(){var t=n.b,r=n.c,o=n.d,a=n.a;return t=t<<25^t>>>7^r,r=r-o|0,o=o<<24^o>>>8^a,a=a-t|0,n.b=t=t<<20^t>>>12^r,n.c=r=r-o|0,n.d=o<<16^r>>>16^a,n.a=a-t|0},n.a=0,n.b=0,n.c=-1640531527,n.d=1367130551,u===Math.floor(u)?(n.a=u/4294967296|0,n.b=u|0):e+=u;for(var i=0;i<e.length+20;i++)n.b^=e.charCodeAt(i)|0,n.next()}function h(u,n){return n.a=u.a,n.b=u.b,n.c=u.c,n.d=u.d,n}function v(u,n){var e=new p(u),i=n&&n.state,t=function(){return(e.next()>>>0)/4294967296};return t.double=function(){do var r=e.next()>>>11,o=(e.next()>>>0)/4294967296,a=(r+o)/(1<<21);while(a===0);return a},t.int32=e.next,t.quick=t,i&&(typeof i=="object"&&h(i,e),t.state=function(){return h(e,{})}),t}s&&s.exports?s.exports=v:c&&c.amd?c(function(){return v}):this.tychei=v})(E,x,!1)})(K);var T={},L={get exports(){return T},set exports(x){T=x}};const Q={},W=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"})),Y=U(W);(function(x){(function(j,s,c){var p=256,h=6,v=52,u="random",n=c.pow(p,h),e=c.pow(2,v),i=e*2,t=p-1,r;function o(f,l,y){var w=[];l=l==!0?{entropy:!0}:l||{};var g=z(S(l.entropy?[f,G(s)]:f??_(),3),w),m=new a(w),X=function(){for(var d=m.g(h),C=n,A=0;d<e;)d=(d+A)*p,C*=p,A=m.g(1);for(;d>=i;)d/=2,C/=2,A>>>=1;return(d+A)/C};return X.int32=function(){return m.g(4)|0},X.quick=function(){return m.g(4)/4294967296},X.double=X,z(G(m.S),s),(l.pass||y||function(d,C,A,$){return $&&($.S&&b($,m),d.state=function(){return b(m,{})}),A?(c[u]=d,C):d})(X,g,"global"in l?l.global:this==c,l.state)}function a(f){var l,y=f.length,w=this,g=0,m=w.i=w.j=0,X=w.S=[];for(y||(f=[y++]);g<p;)X[g]=g++;for(g=0;g<p;g++)X[g]=X[m=t&m+f[g%y]+(l=X[g])],X[m]=l;(w.g=function(d){for(var C,A=0,$=w.i,B=w.j,M=w.S;d--;)C=M[$=t&$+1],A=A*p+M[t&(M[$]=M[B=t&B+C])+(M[B]=C)];return w.i=$,w.j=B,A})(p)}function b(f,l){return l.i=f.i,l.j=f.j,l.S=f.S.slice(),l}function S(f,l){var y=[],w=typeof f,g;if(l&&w=="object")for(g in f)try{y.push(S(f[g],l-1))}catch{}return y.length?y:w=="string"?f:f+"\0"}function z(f,l){for(var y=f+"",w,g=0;g<y.length;)l[t&g]=t&(w^=l[t&g]*19)+y.charCodeAt(g++);return G(l)}function _(){try{var f;return r&&(f=r.randomBytes)?f=f(p):(f=new Uint8Array(p),(j.crypto||j.msCrypto).getRandomValues(f)),G(f)}catch{var l=j.navigator,y=l&&l.plugins;return[+new Date,j,y,j.screen,G(s)]}}function G(f){return String.fromCharCode.apply(0,f)}if(z(c.random(),s),x.exports){x.exports=o;try{r=Y}catch{}}else c["seed"+u]=o})(typeof self<"u"?self:E,[],Math)})(L);var Z=D,nn=O,tn=R,rn=k,en=N,on=P,q=T;q.alea=Z;q.xor128=nn;q.xorwow=tn;q.xorshift7=rn;q.xor4096=en;q.tychei=on;var an=q;export{an as s};
