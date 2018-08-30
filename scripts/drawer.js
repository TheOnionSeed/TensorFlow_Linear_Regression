$(document).ready(init);
function generateData(numPoints, coeff, sigma = 0.3) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];

    const xs = tf.randomUniform([numPoints], -2, 2);

    // Generate polynomial data
    const three = tf.scalar(3, 'int32');
    const ys = a.mul(xs.pow(three))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
	  .add(tf.randomNormal([numPoints], -1*sigma, sigma));;

    return {
      xs, 
      ys
    };
  })
}

var myGraph;
   var randa = Math.random();
   var randb = Math.random();
   var randc = Math.random();
   var randd = Math.random();
   
   const a = tf.variable(tf.scalar(randa));
   const b = tf.variable(tf.scalar(randb));       
   const c = tf.variable(tf.scalar(randc));
  const d = tf.variable(tf.scalar(randd));
   a.print();
   b.print();
   c.print();
   
    
   function predict(x){
      return tf.tidy(() =>{
             // console.log("a:" + a.dataSync()[0]);
             // console.log("b:" + b.dataSync()[0]);
             // console.log("c:" +c.dataSync()[0]);
			 // console.log("d:" +d.dataSync()[0]);
			 
         return a.mul(x.pow(tf.scalar(3))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d);
      });
   }

   function loss(predictions, labels){
      const meanSquareError = predictions.sub(labels).square().mean();
      return meanSquareError;
   }
      
   function train(xs,ys,numIterations=75){
      const learningRate = 0.05;
      const optimizer = tf.train.sgd(learningRate);
      
      for(let i=0; i < numIterations; i++){
          
         optimizer.minimize(() => {
            const predsYs = predict(xs);
            
            return loss(predsYs, ys);
         });
      }
   }
   
async function init(){
	const trueCoefficients = {a: -0.8, b: -0.2, c: 0.9, d: 0.4};
	const trainingData = generateData(100, trueCoefficients);
   myGraph = new Graph({
              canvasId: 'myCanvas',
              minX: -10,
              minY: -10,
              maxX: 10,
              maxY: 10,
              unitsPerTick: 1
            });


   myGraph.drawEquation(function(x) {
      return -0.8*x*x*x -0.2*x*x +0.9*x + 0.4 ;
   }, 'blue', 4);
   
    var coordinatesx = trainingData.xs.dataSync(); 
    console.log(coordinatesx.length);
	
    var coordinatesy = trainingData.ys.dataSync(); 
    console.log(coordinatesy.length);   
	
    for(var i=0; i < coordinatesx.length; i++){
       myGraph.drawCoordinates(coordinatesx[i],coordinatesy[i], 'red', 1);
    }
  
    train(trainingData.xs,trainingData.ys,numIterations=75);

      myGraph.drawEquation(function(x) {
         return a.dataSync()[0]*x*x*x + b.dataSync()[0]*x*x + c.dataSync()[0]*x + d.dataSync()[0] ;
      }, 'green', 2);
      
  
}

