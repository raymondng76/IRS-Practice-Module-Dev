
def create_actor_model(self):
    state_input = Input(shape=self.env.observation_space.shape)
    h1 = Dense(24, activation='relu')(state_input)
    h2 = Dense(48, activation='relu')(h1)
    h3 = Dense(24, activation='relu')(h2)
    output = Dense(self.env.action_space.shape[0],  
        activation='relu')(h3)
    
    model = Model(input=state_input, output=output)
    adam  = Adam(lr=0.001)
    model.compile(loss="mse", optimizer=adam)
    return state_input, model

def create_critic_model(self):
    state_input = Input(shape=self.env.observation_space.shape)
    state_h1 = Dense(24, activation='relu')(state_input)
    state_h2 = Dense(48)(state_h1)
    
    action_input = Input(shape=self.env.action_space.shape)
    action_h1    = Dense(48)(action_input)
    
    merged    = Add()([state_h2, action_h1])
    merged_h1 = Dense(24, activation='relu')(merged)
    output = Dense(1, activation='relu')(merged_h1)
    model  = Model(input=[state_input,action_input], output=output)
    
    adam  = Adam(lr=0.001)
    model.compile(loss="mse", optimizer=adam)
    return state_input, action_input, model

def __init__(self):
    self.actor_state_input, self.actor_model = self.create_actor_model()
    _, self.target_actor_model = self.create_actor_model()
    self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]]) 
        
    actor_model_weights = self.actor_model.trainable_weights
    self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
    grads = zip(self.actor_grads, actor_model_weights)
    self.optimize =  tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
    _, _, self.target_critic_model = self.create_critic_model()
    self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)
        
    # Initialize for later gradient calculations
    self.sess.run(tf.initialize_all_variables())

def train(self):
    batch_size = 32
    if len(self.memory) < batch_size:
        return
    rewards = []
    samples = random.sample(self.memory, batch_size)
    self._train_critic(samples)
    self._train_actor(samples)